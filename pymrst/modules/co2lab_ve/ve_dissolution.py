"""CO2 dissolving into brine, in the VE model (MRST co2lab-ve).

The slowest of the trapping mechanisms and the safest: CO2 that has gone
into solution is denser than the brine around it, so it sinks rather than
rises, and it can never leak. Over the centuries a storage site is assessed
for, it is the mechanism that eventually retires the mobile plume.

It is added here as a third conserved quantity, `rs` - the CO2 dissolved
per unit of brine - carried by the brine and fed by the free CO2 above it:

    brine:      d/dt (b_w s_w pv)             + div(b_w v_w)            = 0
    CO2:        d/dt (b_g s_g pv)             + div(b_g v_g)        + E = 0
    dissolved:  d/dt (rs b_w s_w pv)          + div(rs b_w v_w)     - E = 0

so CO2 leaving the free phase (+E in the gas equation) arrives in the brine
(-E in the dissolved equation) and nothing is created. The rate E is what
the physics is in (MRST's `computeDissolutionTransfer`): dissolution runs at
a rate per unit of plume *interface*, but it cannot dissolve CO2 that is not
there, nor push the brine past its saturation. So

    E = min( dis_rate * poro * area,          how fast it can dissolve
             max(min(demand, supply), 0) )    but no faster than either side

with `demand` the CO2 the brine could still take before reaching dis_max, and
`supply` the free CO2 available (in place, injected, or flowing in).

Rate-driven dissolution only. MRST also offers `dis_rate = 0` for
instantaneous dissolution, which it implements by variable switching (as in
black-oil); that is not ported. A large `dis_rate` approaches it.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.modules.co2lab_ve.ve import VEFluid, VEModel, VEState
from pymrst.utils.adi import (
    ad_max,
    ad_min,
    ad_sum,
    combine_equations,
    init_variables_adi,
    value,
)


@dataclass
class DissolutionFluid(VEFluid):
    """A VE fluid that also dissolves CO2 into the brine."""

    dis_rate: float = 5e-11      # dissolution rate per unit interface area
    dis_max: float = 0.03        # brine saturates at this much CO2


@dataclass
class VEDisState(VEState):
    """VE state with the dissolved CO2 carried along."""

    rs: np.ndarray = None        # dissolved CO2 per unit brine

    def copy(self) -> "VEDisState":
        return VEDisState(pressure=self.pressure.copy(), sG=self.sG.copy(),
                          sGmax=self.sGmax.copy(), qGs=self.qGs.copy(),
                          bhp=self.bhp.copy(), rs=self.rs.copy())


class VEDissolutionModel(VEModel):
    """The VE model with rate-driven dissolution of CO2 into the brine."""

    def __init__(self, ts, rock2d, fluid: DissolutionFluid, **kwargs):
        assert isinstance(fluid, DissolutionFluid), \
            "VEDissolutionModel needs a DissolutionFluid"
        assert fluid.dis_rate > 0, (
            "only rate-driven dissolution is ported; dis_rate = 0 "
            "(instantaneous) needs variable switching, as in black-oil. "
            "A large dis_rate approaches it.")
        super().__init__(ts, rock2d, fluid, **kwargs)

    def initial_state(self, p0: np.ndarray, sg0=0.0, rs0=0.0) -> VEDisState:
        nc = self.ts.g.cells.num
        nw = 1 if self.well_cells.size else 0
        return VEDisState(
            pressure=np.asarray(p0, float).ravel().copy(),
            sG=np.full(nc, float(sg0)), sGmax=np.full(nc, float(sg0)),
            rs=np.full(nc, float(rs0)),
            qGs=np.full(nw, self.well_rate), bhp=np.zeros(nw))

    # -- mass accounting ---------------------------------------------------
    def dissolved_co2(self, st: VEDisState) -> float:
        """Dissolved CO2 as a surface volume (rs * b_w * s_w * pv)."""
        f = self.fluid
        b_w = value(f.b_w(st.pressure))
        return float(self.pv @ (st.rs * b_w * (1.0 - st.sG)))

    def total_co2(self, st: VEDisState) -> float:
        """Free plus dissolved. This is what must be conserved."""
        return self.stored_co2(st) + self.dissolved_co2(st)

    # -- equations ---------------------------------------------------------
    def assemble(self, st0: VEDisState, p, sg, rs, qgs, bhp, dt: float):
        f, H, pv = self.fluid, self.ts.H, self.pv
        h, h_max = f.heights(sg, st0.sGmax, H)
        pc = f.pc(h, p, self.grav)

        b_w, b_g = f.b_w(p), f.b_g(p + pc)
        mob_w = f.kr_w(h, h_max, H) / f.mu_w
        mob_g = f.kr_g(h, H) / f.mu_g
        rho_w = b_w * f.rhoWS
        rho_g = b_g * f.rhoGS

        h0, hmax0 = f.heights(st0.sG, st0.sGmax, H)
        pc0 = value(f.pc(h0, st0.pressure, self.grav))
        b_w0 = value(f.b_w(st0.pressure))
        b_g0 = value(f.b_g(st0.pressure + pc0))

        # phase fluxes, as in the base model
        fluxes, upw_flags = [], []
        for press, mob in ((p, mob_w), (p + pc, mob_g)):
            rho = rho_w if mob is mob_w else rho_g
            pot = self._grad(press) - 0.5 * (
                value(rho)[self.N[:, 0]] + value(rho)[self.N[:, 1]]
            ) * self.gdz
            flag = value(pot) <= 0
            fluxes.append(-self.T * self._upw(flag, mob) * pot)
            upw_flags.append(flag)

        v_w, v_g = fluxes
        flag_w, flag_g = upw_flags
        flux_w = self._upw(flag_w, b_w) * v_w
        flux_g = self._upw(flag_g, b_g) * v_g
        # dissolved CO2 rides along with the brine
        flux_dis = self._upw(flag_w, rs * b_w) * v_w

        sw, sw0 = 1.0 - sg, 1.0 - st0.sG

        water = (pv / dt) * (b_w * sw - b_w0 * sw0) + self._div(flux_w)
        gas = (pv / dt) * (b_g * sg - b_g0 * st0.sG) + self._div(flux_g)
        dis = ((pv / dt) * (rs * b_w * sw - st0.rs * b_w0 * sw0)
               + self._div(flux_dis))

        # -- the transfer rate ---------------------------------------------
        # How much more CO2 the brine could hold, per second, if it went all
        # the way to saturation (and kept whatever the flux brings it).
        demand = ((pv / dt) * (f.dis_max * b_w * sw - st0.rs * b_w0 * sw0)
                  + self._div(flux_dis))

        # How much free CO2 there is to dissolve: what is already in place,
        # plus what is injected, plus what flows in. Cannot dissolve more.
        supply = (pv / dt) * (b_g0 * st0.sG)
        inflow = -self._div(flux_g)                 # net influx of free CO2
        supply = supply + ad_max(inflow, 0.0)
        if self.well_cells.size:
            src = np.zeros(self.ts.g.cells.num)
            src[self.well_cells] = self.well_rate / self.well_cells.size
            supply = supply + src

        # dissolve at the interface rate, but no faster than either side
        basic = f.dis_rate * (pv / H) * b_w
        transfer = ad_max(ad_min(demand, supply), 0.0)
        e = ad_min(basic, transfer)

        gas = gas + e
        dis = dis - e
        # keep the transfer for the sGmax update after convergence
        self._last_e = value(e)
        self._last_bg = value(b_g)

        # -- boundaries and wells, as in the base model ---------------------
        if self.bc_cells.size:
            bc = self.bc_cells
            tb = self.bc_trans
            dpw = self.bc_press - p[bc]
            inflow_w = value(dpw) > 0
            mob_w_bc = np.where(inflow_w, f.krW_max / f.mu_w, 0.0) \
                + (~inflow_w) * mob_w[bc]
            q_w = tb * mob_w_bc * dpw
            water[bc] = water[bc] - b_w[bc] * q_w
            # brine leaving carries its dissolved CO2 out with it; brine
            # arriving from the far field is clean
            rs_bc = (~inflow_w) * rs[bc]
            dis[bc] = dis[bc] - rs_bc * b_w[bc] * q_w

            dpg = self.bc_press - (p + pc)[bc]
            q_g = tb * ((~inflow_w) * mob_g[bc]) * dpg
            gas[bc] = gas[bc] - b_g[bc] * q_g

        ctrls = []
        if self.well_cells.size:
            wc = self.well_cells
            wi = self.well_WI
            dd = -bhp[0:1] + p[wc]
            mt = mob_w[wc] + mob_g[wc]
            cq_g = -(wi * mt) * dd
            gas[wc] = gas[wc] - b_g[wc] * cq_g
            ctrls.append(qgs[0:1] - ad_sum(b_g[wc] * cq_g))
            ctrls.append(qgs[0:1] - self.well_rate)

        return [water, gas, dis] + ctrls

    def _update_sgmax(self, st0: VEDisState, st: VEDisState,
                      dt: float) -> np.ndarray:
        """sGmax after dissolution has eaten into the residual CO2.

        Without dissolution the historical maximum only ever grows, and
        sGmax = max(sGmax0, sG). But the CO2 held residually behind the plume
        dissolves too, so the maximum must be allowed to *fall*: MRST solves
        sGmax = max(sGmax0 - s_depletion dt fac, sG). Leave this out and the
        inventory credits residual CO2 that dissolution has already removed -
        the trapping inventory's own mass balance catches it.
        """
        f = self.fluid
        if f.srg <= 0.0:
            return np.maximum(st0.sGmax, st.sG)
        decrement = (self._last_e * dt * (1.0 - f.srw)
                     / (f.srg * self.pv * self._last_bg))
        return np.maximum(st0.sGmax - decrement, st.sG)

    def step(self, state: VEDisState, dt: float
             ) -> Tuple[VEDisState, bool, int]:
        st0, st = state, state.copy()
        nc = self.ts.g.cells.num

        for it in range(self.max_iterations):
            p, sg, rs, qgs, bhp = init_variables_adi(
                st.pressure, st.sG, st.rs, st.qGs, st.bhp)
            eqs = self.assemble(st0, p, sg, rs, qgs, bhp, dt)
            res = np.concatenate([np.atleast_1d(value(e)) for e in eqs])
            scale = max(np.abs(self.pv / dt).max(), 1.0)
            if np.max(np.abs(res)) <= self.tol * scale:
                st.sGmax = self._update_sgmax(st0, st, dt)
                return st, True, it

            eq = combine_equations(*eqs)
            upd = spla.spsolve(eq.jac[0].tocsc(), -eq.val)
            st.pressure = st.pressure + upd[:nc]
            st.sG = np.clip(st.sG + np.clip(upd[nc:2 * nc], -0.2, 0.2),
                            0.0, 1.0 - self.fluid.srw)
            # Limit the rs step. The transfer rate is a min of a max of a
            # min, so its derivative jumps when a clamp switches; without a
            # limit Newton oscillates between the two branches instead of
            # settling on one.
            d_rs = np.clip(upd[2 * nc:3 * nc],
                           -0.25 * self.fluid.dis_max,
                           0.25 * self.fluid.dis_max)
            st.rs = np.clip(st.rs + d_rs, 0.0, self.fluid.dis_max)
            if self.well_cells.size:
                st.qGs = st.qGs + upd[3 * nc:3 * nc + 1]
                st.bhp = st.bhp + upd[3 * nc + 1:3 * nc + 2]

        st.sGmax = self._update_sgmax(st0, st, dt)
        return st, False, self.max_iterations


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.co2lab_mit.inventory import (
        inventory_history,
        mass_trapping_distribution,
    )
    from pymrst.modules.co2lab_spillpoint.trapping import trap_analysis
    from pymrst.modules.co2lab_ve.ve import average_rock, top_surface_grid
    from pymrst.params.gravity_const import NORM_GRAVITY
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import DARCY, DAY, MILLI, YEAR

    def _case(dis_rate=5e-11, dis_max=0.03):
        nx, ny, nz = 40, 16, 3
        g = cart_grid([nx, ny, nz], [2000.0, 800.0, 20.0])
        x, y = g.nodes.coords[:, 0], g.nodes.coords[:, 1]
        g.nodes.coords[:, 2] += (
            1000.0 - 0.02 * x
            - 16.0 * np.exp(-(((x - 1400.0) / 160.0) ** 2
                              + ((y - 400.0) / 220.0) ** 2)))
        g = compute_geometry(g)
        rock = make_rock(g, 300 * MILLI * DARCY, 0.22)
        ts = top_surface_grid(g)
        rock2 = average_rock(rock, ts)

        fluid = DissolutionFluid(
            rhoWS=1020.0, rhoGS=700.0, mu_w=8e-4, mu_g=6e-5,
            srw=0.2, srg=0.2, krW_max=1.0, krG_max=1.0,
            c_water=4.3e-10, c_gas=1e-9, p_ref=1.2e7,
            dis_rate=dis_rate, dis_max=dis_max)
        return ts, rock2, fluid

    def _run(ts, rock2, fluid, n_inj=10, n_mig=40):
        inj = int(np.argmin(ts.g.cells.centroids[:, 0]
                            + 1e3 * np.abs(ts.g.cells.centroids[:, 1] - 400.0)))
        rate = 0.05e9 / YEAR / fluid.rhoGS
        model = VEDissolutionModel(ts, rock2, fluid,
                                   well_cells=np.array([inj]),
                                   well_rate=rate, well_WI=np.array([2e-11]))
        p0 = 1e5 + fluid.rhoWS * NORM_GRAVITY * ts.z
        st = model.initial_state(p0)
        st.bhp = np.array([p0[inj] + 5e5])

        states = [st]
        for k in range(n_inj):
            st, ok, _ = model.step(st, 60.0 * DAY)
            assert ok, f"injection step {k} failed"
            states.append(st)

        # Quarter-year steps. The transfer rate is a min of a max of a min,
        # so its derivative jumps when a clamp switches; Newton chatters on
        # that at a half-year step, where the base VE model - which has no
        # such term - happily takes a full year.
        quiet = VEDissolutionModel(ts, rock2, fluid)
        st = VEDisState(pressure=st.pressure, sG=st.sG, sGmax=st.sGmax,
                        rs=st.rs, qGs=np.zeros(0), bhp=np.zeros(0))
        for k in range(n_mig):
            st, ok, _ = quiet.step(st, 0.25 * YEAR)
            assert ok, f"migration step {k} failed"
            states.append(st)
        return model, quiet, states, n_inj

    def test_dissolution_conserves_total_co2():
        """CO2 leaving the free phase must arrive in the brine, exactly.

        The whole point of writing the transfer as +E in one equation and -E
        in the other. Free CO2 falls, dissolved CO2 rises, and the sum is
        untouched once injection stops.
        """
        ts, rock2, fluid = _case()
        model, quiet, states, n_inj = _run(ts, rock2, fluid)

        free = np.array([quiet.stored_co2(s) for s in states[n_inj:]])
        diss = np.array([quiet.dissolved_co2(s) for s in states[n_inj:]])
        total = free + diss

        assert diss[-1] > 0.0, "nothing dissolved at all"
        assert diss[-1] > diss[0], "dissolved CO2 should grow with time"
        assert free[-1] < free[0], "free CO2 should be eaten away"

        drift = (total.max() - total.min()) / total[0]
        assert drift < 1e-4, f"total CO2 drifted by {drift:.2e}"
        print(f"ok test_dissolution_conserves_total_co2 "
              f"(free {free[0] / 1e6:.2f} -> {free[-1] / 1e6:.2f} Mm3, "
              f"dissolved 0 -> {diss[-1] / 1e6:.2f} Mm3; "
              f"total conserved to {drift:.1e})")

    def test_brine_saturates_at_dis_max():
        """rs can never exceed dis_max, however long you wait.

        The demand clamp. Run with an enormous rate so the limit is the
        brine's capacity, not the kinetics.
        """
        ts, rock2, fluid = _case(dis_rate=1e-6, dis_max=0.02)
        _, quiet, states, _ = _run(ts, rock2, fluid, n_mig=40)

        st = states[-1]
        assert st.rs.max() <= fluid.dis_max + 1e-12, st.rs.max()
        # and somewhere under the plume the brine really is saturated
        assert st.rs.max() > 0.9 * fluid.dis_max, (
            f"with a huge rate the brine should saturate "
            f"(got {st.rs.max():.4f} of {fluid.dis_max})")
        # brine the plume never touched stays clean
        assert st.rs.min() < 1e-6
        print(f"ok test_brine_saturates_at_dis_max "
              f"(rs reaches {st.rs.max():.4f}, capped at {fluid.dis_max})")

    def test_rate_controls_how_fast_it_dissolves():
        """A faster dis_rate dissolves more in the same time - up to the cap."""
        out = {}
        for rate in (5e-12, 5e-11, 5e-10):
            ts, rock2, fluid = _case(dis_rate=rate)
            _, quiet, states, n_inj = _run(ts, rock2, fluid, n_mig=30)
            out[rate] = quiet.dissolved_co2(states[-1])

        vals = [out[r] for r in (5e-12, 5e-11, 5e-10)]
        assert vals[0] < vals[1] < vals[2], (
            f"dissolution should speed up with dis_rate: {vals}")
        print(f"ok test_rate_controls_how_fast_it_dissolves "
              f"(dissolved {vals[0] / 1e6:.3f}, {vals[1] / 1e6:.3f}, "
              f"{vals[2] / 1e6:.3f} Mm3 for rate 5e-12, 5e-11, 5e-10)")

    def test_cannot_dissolve_co2_that_is_not_there():
        """The supply clamp: no free CO2, no dissolution.

        Brine with no plume above it must stay clean no matter how large the
        rate. Without the supply clamp the rate law would happily invent CO2.
        """
        ts, rock2, fluid = _case(dis_rate=1e-5)
        model = VEDissolutionModel(ts, rock2, fluid)
        p0 = 1e5 + fluid.rhoWS * NORM_GRAVITY * ts.z
        st = model.initial_state(p0, sg0=0.0, rs0=0.0)   # no CO2 anywhere

        for _ in range(5):
            st, ok, _ = model.step(st, 1.0 * YEAR)
            assert ok
        assert np.all(st.rs <= 1e-12), (
            f"CO2 dissolved out of nothing (rs up to {st.rs.max():.2e})")
        assert model.total_co2(st) < 1e-6
        print("ok test_cannot_dissolve_co2_that_is_not_there")

    def test_inventory_dissolved_bar_fills_up():
        """The inventory's `dissolved` category now has something in it.

        This is what the whole exercise was for: before dissolution the first
        bar of the trapping inventory was identically zero.
        """
        ts, rock2, fluid = _case(dis_rate=5e-11)
        model, quiet, states, n_inj = _run(ts, rock2, fluid, n_mig=40)
        res = trap_analysis(ts)

        end = states[-1]
        # feed the model's own rs into the inventory
        inv0 = mass_trapping_distribution(ts, states[n_inj], rock2, fluid, res,
                                          rs=float(states[n_inj].rs.mean()))
        inv1 = mass_trapping_distribution(ts, end, rock2, fluid, res,
                                          rs=float(end.rs.mean()))

        assert inv1.masses[0] > 0.0, "the dissolved bar is still empty"
        assert inv1.masses[0] > inv0.masses[0], "dissolution should grow"
        # and the mobile fraction has fallen
        assert inv1.mobile / inv1.free_phase < inv0.mobile / inv0.free_phase
        print(f"ok test_inventory_dissolved_bar_fills_up "
              f"(dissolved {inv1.masses[0] / 1e9:.4f} Mt; mobile fraction "
              f"{inv0.mobile / inv0.free_phase:.0%} -> "
              f"{inv1.mobile / inv1.free_phase:.0%})")

    test_cannot_dissolve_co2_that_is_not_there()
    test_dissolution_conserves_total_co2()
    test_brine_saturates_at_dis_max()
    test_rate_controls_how_fast_it_dissolves()
    test_inventory_dissolved_bar_fills_up()
    print("\nAll tests passed!")
