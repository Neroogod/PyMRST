"""CO2-brine two-phase model (MRST co2lab: TwoPhaseWaterGasModel).

Fully-implicit, compressible, buoyancy-driven water/gas flow:

    (pv/dt)(b_ph s_ph - b_ph0 s_ph0) + div(b_ph,upw v_ph) = q_ph
    v_ph = -(T/mu_ph) kr_ph (grad p - rho_ph g grad z)

with CO2 density and viscosity taken from the sampled property tables at
a fixed reservoir temperature (addSampledFluidProperties), and brine of
constant compressibility. Primary variables are (p, sG) plus (qWs, qGs,
bhp) per well. This is the workhorse for CO2 storage and, with hydrogen
properties substituted, for H2 storage.
"""

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.modules.co2lab_common.co2props import CO2Props, co2_props
from pymrst.params.gravity_const import NORM_GRAVITY
from pymrst.params.rock import Rock, pore_volume
from pymrst.params.wells_and_bc import Well
from pymrst.utils.adi import (
    ad_exp,
    ad_max,
    ad_min,
    ad_sum,
    combine_equations,
    init_variables_adi,
    value,
    vertcat_adi,
)


def _clamp01(s):
    """Clamp a saturation to [0, 1], carrying derivatives through."""
    return ad_min(ad_max(s, 0.0), 1.0)


@dataclass
class WaterGasFluid:
    """Brine + CO2 properties at a fixed reservoir temperature."""

    rhoWS: float                 # brine surface density, kg/m3
    rhoGS: float                 # CO2 surface (reference) density, kg/m3
    b_w: Callable                # bW(p)
    b_g: Callable                # bG(p)
    mu_w: Callable
    mu_g: Callable
    kr_w: Callable
    kr_g: Callable


def co2_brine_fluid(
    temperature: float,
    rhoWS: float = 1020.0,
    c_water: float = 4.3e-10,
    p_ref: float = 1e7,
    mu_water: float = 8e-4,
    srw: float = 0.0,
    src: float = 0.0,
    n: Tuple[float, float] = (2.0, 2.0),
    props: Optional[CO2Props] = None,
    rho_gs: Optional[float] = None,
) -> WaterGasFluid:
    """Brine/CO2 fluid; CO2 density and viscosity come from the tables.

    rho_gs defaults to the CO2 density at surface conditions used by
    co2lab (1 atm, the given temperature is not applied at the surface;
    MRST uses the density at reference pressure/temperature as rhoGS).
    """
    if props is None:
        props = co2_props()
    t = float(temperature)

    if rho_gs is None:
        rho_gs = float(props.rho(np.array([1.01325e5]), np.array([t]))[0])

    def b_w(p):
        return ad_exp(c_water * (p - p_ref))

    def b_g(p):
        return props.rho(p, np.full(len(value(p)), t)) / rho_gs

    def mu_g(p):
        return props.mu(p, np.full(len(value(p)), t))

    den = 1.0 - srw - src

    def kr_w(sw):
        # normalized saturation, clamped to [0, 1] (AD-safe)
        return _clamp01((sw - srw) / den) ** n[0]

    def kr_g(sg):
        return _clamp01((sg - src) / den) ** n[1]

    return WaterGasFluid(
        rhoWS=rhoWS, rhoGS=rho_gs,
        b_w=b_w, b_g=b_g,
        mu_w=lambda p: mu_water + 0.0 * value(p),
        mu_g=mu_g, kr_w=kr_w, kr_g=kr_g,
    )


@dataclass
class WGState:
    """State of the CO2-brine model."""

    pressure: np.ndarray
    sG: np.ndarray
    qWs: np.ndarray
    qGs: np.ndarray
    bhp: np.ndarray

    def copy(self) -> "WGState":
        return WGState(*(v.copy() for v in (self.pressure, self.sG,
                                            self.qWs, self.qGs, self.bhp)))


class CO2BrineModel:
    """Fully-implicit two-phase water/gas model with gravity."""

    def __init__(
        self,
        g: Grid,
        rock: Rock,
        fluid: WaterGasFluid,
        wells: Sequence[Well],
        gravity: float = NORM_GRAVITY,
        ds_max: float = 0.2,
        tol: float = 1e-9,
        max_iterations: int = 30,
    ):
        self.g = g
        self.rock = rock
        self.fluid = fluid
        self.wells = list(wells)
        self.ops = setup_operators_tpfa(g, rock)
        self.pv = pore_volume(g, rock)
        self.gdz = gravity * self.ops.grad(g.cells.centroids[:, 2])
        self.grav = gravity
        self.ds_max = ds_max
        self.tol = tol
        self.max_iterations = max_iterations

    def initial_state(self, p0: np.ndarray, sg0: np.ndarray) -> WGState:
        nw = len(self.wells)
        bhp = np.array([w.val if w.type == "bhp" else float(p0[w.cells[0]])
                        for w in self.wells])
        qws = np.zeros(nw)
        qgs = np.array([w.val if w.type == "rate" else 0.0
                        for w in self.wells])
        return WGState(pressure=np.asarray(p0, float).copy(),
                       sG=np.asarray(sg0, float).copy(),
                       qWs=qws, qGs=qgs, bhp=bhp)

    def assemble(self, st0: WGState, p, sg, qws, qgs, bhp, dt: float):
        f, ops, pv = self.fluid, self.ops, self.pv
        sw = 1.0 - sg
        b_w, b_g = f.b_w(p), f.b_g(p)
        mob_w = f.kr_w(sw) / f.mu_w(p)
        mob_g = f.kr_g(sg) / f.mu_g(p)
        rho_w = b_w * f.rhoWS
        rho_g = b_g * f.rhoGS

        b_w0 = value(f.b_w(st0.pressure))
        b_g0 = value(f.b_g(st0.pressure))

        eqs = []
        upw_flags = []
        for mob, rho, b in ((mob_w, rho_w, b_w), (mob_g, rho_g, b_g)):
            pot = ops.grad(p) - ops.avg(rho) * self.gdz
            flag = value(pot) <= 0
            upw_flags.append(flag)
            v = -ops.T * ops.upw(flag, mob) * pot
            eqs.append(ops.div(ops.upw(flag, b) * v))

        water = (pv / dt) * (b_w * sw - b_w0 * (1.0 - st0.sG)) + eqs[0]
        gas = (pv / dt) * (b_g * sg - b_g0 * st0.sG) + eqs[1]

        ctrls = []
        for k, w in enumerate(self.wells):
            wc = w.cells
            # hydrostatic connection pressure drop with the injected phase
            rho_inj = value(rho_g)[wc] if w.compi[1] > 0.5 \
                else value(rho_w)[wc]
            cdp = self.grav * w.dZ * rho_inj
            dd = -(bhp[k:k + 1] + cdp) + p[wc]
            inj = value(dd) < 0
            wi = w.WI
            mt = mob_w[wc] + mob_g[wc]
            # injection: total mobility; production: phase mobilities
            cq_w = -(wi * (~inj)) * mob_w[wc] * dd \
                + -(wi * inj) * mt * dd * w.compi[0]
            cq_g = -(wi * (~inj)) * mob_g[wc] * dd \
                + -(wi * inj) * mt * dd * w.compi[1]
            cq_ws = b_w[wc] * cq_w
            cq_gs = b_g[wc] * cq_g

            water[wc] = water[wc] - cq_ws
            gas[wc] = gas[wc] - cq_gs
            ctrls.append(qws[k:k + 1] - ad_sum(cq_ws))
            ctrls.append(qgs[k:k + 1] - ad_sum(cq_gs))
            if w.type == "bhp":
                ctrls.append(bhp[k:k + 1] - w.val)
            elif w.type == "rate":
                ctrls.append(qws[k:k + 1] + qgs[k:k + 1] - w.val)
            else:
                raise ValueError(f"unsupported control '{w.type}'")

        return [water, gas] + ctrls

    def step(self, state: WGState, dt: float) -> Tuple[WGState, bool, int]:
        st0 = state
        st = state.copy()
        nc, nw = self.g.cells.num, len(self.wells)

        for it in range(self.max_iterations):
            p, sg, qws, qgs, bhp = init_variables_adi(
                st.pressure, st.sG, st.qWs, st.qGs, st.bhp
            )
            eqs = self.assemble(st0, p, sg, qws, qgs, bhp, dt)
            res = np.concatenate([np.atleast_1d(value(e)) for e in eqs])
            scale = max(np.abs(self.pv / dt).max(), 1.0)
            if np.max(np.abs(res)) <= self.tol * scale:
                return st, True, it
            eq = combine_equations(*eqs)
            upd = spla.spsolve(eq.jac[0].tocsc(), -eq.val)

            st.pressure = st.pressure + upd[:nc]
            ds = np.clip(upd[nc:2 * nc], -self.ds_max, self.ds_max)
            st.sG = np.clip(st.sG + ds, 0.0, 1.0)
            off = 2 * nc
            st.qWs = st.qWs + upd[off:off + nw]
            st.qGs = st.qGs + upd[off + nw:off + 2 * nw]
            st.bhp = st.bhp + upd[off + 2 * nw:]

        return st, False, self.max_iterations


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.utils.units import BARSA, DARCY, DAY, MILLI, YEAR

    MASS_RATE = 0.1e9 / YEAR      # 0.1 Mt CO2 per year, kg/s

    def _aquifer():
        """2 km long, 50 m thick aquifer whose top lies at 1200 m depth."""
        g = cart_grid([20, 1, 10], [2000.0, 100.0, 50.0])
        g.nodes.coords[:, 2] += 1200.0
        return g

    def build():
        """A 2 km long, 50 m thick aquifer at 1200 m; CO2 injected at the
        bottom of the left edge."""
        g = compute_geometry(_aquifer())
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = co2_brine_fluid(temperature=273.15 + 45.0, srw=0.2, src=0.05)
        # bottom-left cell (k = 9)
        inj = 0 + 20 * 1 * 9
        rate = MASS_RATE / fluid.rhoGS          # surface volume rate, m3/s
        w = add_well(None, g, rock, np.array([inj]), well_type="rate",
                     val=rate, radius=0.15,
                     compi=np.array([0.0, 1.0]), sign=1,
                     gravity=np.array([0.0, 0.0, NORM_GRAVITY]))
        model = CO2BrineModel(g, rock, fluid, w)
        p0 = 1e5 + 1000.0 * NORM_GRAVITY * g.cells.centroids[:, 2]
        st = model.initial_state(p0, np.zeros(g.cells.num))
        return g, model, st, rate

    def test_co2_density_at_reservoir_conditions():
        """CO2 is supercritical and lighter than brine at 1200 m / 45 C."""
        f = co2_brine_fluid(temperature=273.15 + 45.0)
        p = np.array([120 * BARSA])
        rho_co2 = float(f.b_g(p)[0] * f.rhoGS)
        assert 600.0 < rho_co2 < 800.0, rho_co2
        assert rho_co2 < f.rhoWS, "CO2 must be buoyant in brine"
        print(f"ok test_co2_density (rho_CO2 = {rho_co2:.1f} kg/m3)")

    def test_injection_converges_and_conserves():
        """CO2 injection converges and the surface volume balances."""
        g, model, st, rate = build()
        f, pv = model.fluid, model.pv
        gas0 = float(np.sum(pv * value(f.b_g(st.pressure)) * st.sG))
        dts = np.full(6, 30.0 * DAY)
        rates = []
        for dt in dts:
            st, ok, its = model.step(st, dt)
            assert ok, f"Newton failed after {its} iterations"
            rates.append(st.qGs[0])
        gas1 = float(np.sum(pv * value(f.b_g(st.pressure)) * st.sG))
        injected = float(np.sum(np.asarray(rates) * dts))
        assert np.isclose(gas1 - gas0, injected, rtol=1e-6), (
            f"gas gained {gas1 - gas0:.4e} vs injected {injected:.4e}"
        )
        assert np.isclose(st.qGs[0], rate, rtol=1e-8)
        print("ok test_injection_converges_and_conserves")

    def test_buoyant_migration():
        """A CO2 slug released at the bottom of the aquifer migrates to the
        caprock, leaving residually trapped CO2 behind."""
        g = compute_geometry(_aquifer())
        rock = make_rock(g, 100 * MILLI * DARCY, 0.25)
        fluid = co2_brine_fluid(temperature=273.15 + 45.0, srw=0.2, src=0.05)
        model = CO2BrineModel(g, rock, fluid, [])
        z = g.cells.centroids[:, 2]
        pv = model.pv

        p0 = 1e5 + fluid.rhoWS * NORM_GRAVITY * z
        sg0 = np.where(z > z.max() - 10.0, 0.4, 0.0)   # slug at the bottom
        st = WGState(pressure=p0, sG=sg0, qWs=np.zeros(0),
                     qGs=np.zeros(0), bhp=np.zeros(0))

        top = z < z.min() + 5.0
        bot = z > z.max() - 5.0
        co2 = lambda s, sel: float(pv[sel] @ s.sG[sel])
        top0, bot0 = co2(st, top), co2(st, bot)

        for _ in range(10):
            st, ok, _ = model.step(st, 180.0 * DAY)
            assert ok, "migration step failed"

        assert co2(st, top) > 10.0 * max(top0, 1.0), "CO2 must reach the top"
        assert co2(st, bot) < 0.5 * bot0, "the source layer must drain"
        # residual trapping: CO2 left behind is at (or below) s_gr
        mid = ~top & ~bot
        assert np.all(st.sG[mid] <= 0.4 + 1e-9)
        assert st.sG[top].max() > 0.3, "a mobile plume must pool at the top"
        print(f"ok test_buoyant_migration (top CO2 {top0:.0f} -> "
              f"{co2(st, top):.0f} m3)")

    test_co2_density_at_reservoir_conditions()
    test_injection_converges_and_conserves()
    test_buoyant_migration()
    print("\nAll tests passed!")
