"""Single-phase geothermal flow (MRST geothermal/models/GeothermalModel).

Mass and energy, solved together. Two things make this more than a
compressible water simulation:

  - the fluid properties depend on temperature, strongly. Cold water is
    denser and six times more viscous than hot, so injecting cold water
    changes the flow field it is injected into.

  - heat moves two ways: it is *carried* by the water (advection) and it
    *diffuses* through the rock (conduction). The rock is the larger heat
    store by far, which is why a thermal front crawls along at a small
    fraction of the water's speed.

    mass:    d/dt (pv rho) + div(rho v)                        = q
    energy:  d/dt (pv rho u + (V - pv) rho_R Cp_R T)
                 + div(h rho v) - div((Tr + Tf) grad T)        = q_e

with u = Cp T and h = u + p/rho, exactly as MRST defines them. The heat
transmissibilities Tr and Tf come from the rock and fluid conductivities
weighted by (1 - poro) and poro.
"""

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.modules.geothermal.water_props import (
    density_brine,
    viscosity_brine,
)
from pymrst.params.rock import Rock, make_rock, pore_volume
from pymrst.params.wells_and_bc import BoundaryCondition
from pymrst.solvers.trans import compute_trans
from pymrst.utils.adi import combine_equations, init_variables_adi, value

# MRST's defaults (addThermalRockProps / addThermalFluidProps)
LAMBDA_R = 2.0          # rock thermal conductivity, W/(m K)
LAMBDA_F = 0.6          # fluid thermal conductivity, W/(m K)
RHO_R = 2700.0          # rock density, kg/m3
CP_R = 1000.0           # rock heat capacity, J/(kg K)
CP_W = 4200.0           # water heat capacity, J/(kg K)


def _field(obj, name, default=None):
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


@dataclass
class ThermalFluid:
    """Water or brine with the Spivey/Kestin EOS."""

    salinity: float = 0.0          # NaCl mass fraction
    Cp: float = CP_W
    lambda_f: float = LAMBDA_F

    def rho(self, p, t):
        return density_brine(p, t, self.salinity)

    def mu(self, p, t):
        return viscosity_brine(p, t, self.salinity)

    def u(self, p, t):
        """Internal energy, u = Cp T."""
        return self.Cp * t

    def h(self, p, t):
        """Enthalpy, h = u + p / rho."""
        return self.u(p, t) + p / self.rho(p, t)


@dataclass
class ThermalRock:
    """Rock with its heat capacity and conductivity."""

    lambda_r: float = LAMBDA_R
    rho_r: float = RHO_R
    Cp_r: float = CP_R


@dataclass
class GeothermalState:
    """Pressure and temperature."""

    pressure: np.ndarray
    T: np.ndarray

    def copy(self) -> "GeothermalState":
        return GeothermalState(self.pressure.copy(), self.T.copy())


class GeothermalModel:
    """Fully-implicit single-phase mass and energy (GeothermalModel)."""

    def __init__(
        self,
        g: Grid,
        rock: Rock,
        fluid: ThermalFluid,
        thermal_rock: ThermalRock = ThermalRock(),
        bc: Optional[BoundaryCondition] = None,
        bc_T: Optional[np.ndarray] = None,
        wells: Sequence[Any] = (),
        well_T: Optional[np.ndarray] = None,
        src: Optional[Any] = None,
        src_T: Optional[np.ndarray] = None,
        tol: float = 1e-8,
        max_iterations: int = 30,
    ):
        self.g = g
        self.rock = rock
        self.fluid = fluid
        self.trock = thermal_rock
        self.bc = bc
        self.bc_T = None if bc_T is None else np.asarray(bc_T, float).ravel()
        # Wells and sources carry an *injection* temperature each. It is
        # required rather than defaulted, for the same reason `bc_T` is: water
        # injected at an unstated temperature would silently arrive at the
        # reservoir's own, and the whole point of a thermal model is that it
        # does not.
        self.wells = list(wells)
        self.well_T = None if well_T is None else np.asarray(well_T, float).ravel()
        self.src = src
        self.src_T = None if src_T is None else np.asarray(src_T, float).ravel()

        self.ops = setup_operators_tpfa(g, rock)
        self.pv = pore_volume(g, rock)
        self.vol = g.cells.volumes
        poro = np.asarray(rock.poro, float).ravel()

        # heat transmissibilities: the same TPFA harmonic average, with the
        # volume-weighted conductivities standing in for permeability
        lam_r = thermal_rock.lambda_r * (1.0 - poro)
        lam_f = fluid.lambda_f * poro
        self.T_heat = (self._face_trans(lam_r) + self._face_trans(lam_f))
        self.tol = tol
        self.max_iterations = max_iterations

        if self.wells and self.well_T is None:
            raise ValueError(
                "a thermal well needs an injection temperature (well_T, one "
                "per well); without it injected water arrives at the "
                "reservoir temperature and heats nothing")
        if src is not None and self.src_T is None:
            raise ValueError(
                "a thermal source needs an injection temperature (src_T)")

        if bc is not None and bc.face.size:
            assert self.bc_T is not None, \
                "a pressure bc needs a boundary temperature (bc_T)"
            assert all(t == "pressure" for t in bc.type)
            t_half = compute_trans(g, rock)
            self.t_all = 1.0 / np.bincount(g.cells.faces, weights=1.0 / t_half,
                                           minlength=g.faces.num)
            self.th_all = self._harmonic(self._half_trans(lam_r)
                                         + self._half_trans(lam_f))
            faces = np.asarray(bc.face, np.int64).ravel()
            nb = g.faces.neighbors[faces]
            self.bc_faces = faces
            self.bc_cells = nb.max(axis=1)

    def _half_trans(self, lam: np.ndarray) -> np.ndarray:
        rock = make_rock(self.g, lam[:, None],
                         np.asarray(self.rock.poro, float).reshape(-1, 1))
        return compute_trans(self.g, rock)

    def _harmonic(self, th: np.ndarray) -> np.ndarray:
        """Half-face -> face transmissibility. Zero conductivity gives zero,
        not a division by zero: switching conduction off is a legitimate way
        to isolate advection."""
        with np.errstate(divide="ignore"):
            inv = np.where(th > 0, 1.0 / np.where(th > 0, th, 1.0), np.inf)
        tot = np.bincount(self.g.cells.faces, weights=inv,
                          minlength=self.g.faces.num)
        return np.where(np.isfinite(tot) & (tot > 0), 1.0 / np.where(tot > 0, tot, 1.0), 0.0)

    def _face_trans(self, lam: np.ndarray) -> np.ndarray:
        t_face = self._harmonic(self._half_trans(lam))
        internal = (self.g.faces.neighbors >= 0).all(axis=1)
        return t_face[internal]

    def initial_state(self, p0: float, t0: float) -> GeothermalState:
        nc = self.g.cells.num
        return GeothermalState(pressure=np.full(nc, float(p0)),
                               T=np.full(nc, float(t0)))

    def energy(self, p, t):
        """Total thermal energy per cell: fluid plus rock."""
        f, tr = self.fluid, self.trock
        fluid_e = self.pv * f.rho(p, t) * f.u(p, t)
        rock_e = (self.vol - self.pv) * tr.rho_r * tr.Cp_r * t
        return fluid_e + rock_e

    def assemble(self, st0: GeothermalState, p, t, dt: float):
        f, ops = self.fluid, self.ops

        rho = f.rho(p, t)
        rho0 = f.rho(st0.pressure, st0.T)
        mob = 1.0 / f.mu(p, t)
        h = f.h(p, t)

        pot = ops.grad(p)
        flag = value(pot) <= 0
        v = -ops.T * ops.upw(flag, mob) * pot
        mass_flux = ops.upw(flag, rho) * v                # rho v

        mass = (self.pv / dt) * (rho - rho0) + ops.div(mass_flux)

        # energy: what the water carries, plus what conducts through the rock
        e = self.energy(p, t)
        e0 = self.energy(st0.pressure, st0.T)
        advective = ops.upw(flag, h) * mass_flux
        conductive = -self.T_heat * ops.grad(t)
        energy = (e - e0) / dt + ops.div(advective) + ops.div(conductive)

        if self.bc is not None and self.bc.face.size:
            c = self.bc_cells
            p_bc = np.asarray(self.bc.value, float).ravel()
            t_bc = self.bc_T
            dp = p_bc - p[c]
            inflow = value(dp) > 0
            fin = inflow.astype(np.float64)

            # inflowing water arrives at the boundary temperature; outflowing
            # water leaves at the cell's own
            q = rho[c] * mob[c] * self.t_all[self.bc_faces] * dp
            h_bc = f.h(p_bc, t_bc)
            h_up = fin * h_bc + (1.0 - fin) * h[c]

            mass[c] = mass[c] - q
            # advected heat, plus conduction across the boundary face
            q_cond = self.th_all[self.bc_faces] * (t_bc - t[c])
            energy[c] = energy[c] - h_up * q - q_cond

        mass, energy = self._add_sources(mass, energy, p, t, rho, mob, h)
        mass, energy = self._add_wells(mass, energy, p, t, rho, mob, h)
        return mass, energy

    def _add_sources(self, mass, energy, p, t, rho, mob, h):
        """Prescribed cell rates and the heat they carry.

        Same upwinding as the boundary block above: water *entering* the
        reservoir arrives at the source's own temperature, water leaving takes
        the cell's. A source is a rate, so there is no mobility term - it is
        prescribed, not driven by a pressure difference.
        """
        if self.src is None:
            return mass, energy
        cells = np.asarray(_field(self.src, "cell"), dtype=int).ravel()
        rates = np.asarray(_field(self.src, "rate"), dtype=float).ravel()
        if not cells.size:
            return mass, energy

        injecting = (rates > 0).astype(float)
        h_src = self.fluid.h(p[cells], self.src_T)
        h_up = injecting*h_src + (1.0 - injecting)*h[cells]
        q = rho[cells]*rates

        mass[cells] = mass[cells] - q
        energy[cells] = energy[cells] - h_up*q
        return mass, energy

    def _add_wells(self, mass, energy, p, t, rho, mob, h):
        """Peaceman well rates and the heat they carry.

        The structure is the boundary block's, with the well index in place of
        the face transmissibility and the well's bottom-hole pressure in place
        of the boundary pressure:

            q = rho * mob * WI * (p_bh - p_cell)

        and the enthalpy is upwinded the same way - an *injecting* perforation
        carries the well's injection enthalpy, a *producing* one carries the
        cell's. Using the cell's enthalpy for an injector is the error this
        exists to avoid: cold injected water would arrive at reservoir
        temperature and the thermal front would never form.

        Only bottom-hole-pressure control is handled. A rate-controlled well
        needs its own unknown and a control equation, which is a well *model*
        rather than a source term, and inventing a bottom-hole pressure for it
        would put the wrong drawdown on every perforation.
        """
        if not self.wells:
            return mass, energy

        for index, well in enumerate(self.wells):
            control = str(getattr(well, "type", "bhp")).lower()
            if control != "bhp":
                raise NotImplementedError(
                    f"thermal well {index} is on {control!r} control; the "
                    f"geothermal kernel handles bottom-hole pressure only - a "
                    f"rate control needs a well unknown and a closure "
                    f"equation, not a source term")
            cells = np.atleast_1d(np.asarray(well.cells, dtype=int)).ravel()
            wi = np.atleast_1d(np.asarray(well.WI, dtype=float)).ravel()
            p_bh = float(getattr(well, "val", 0.0))

            dp = p_bh - p[cells]
            injecting = (value(dp) > 0).astype(float)
            h_well = self.fluid.h(np.full(cells.size, p_bh),
                                  np.full(cells.size, float(self.well_T[index])))
            h_up = injecting*h_well + (1.0 - injecting)*h[cells]

            q = rho[cells]*mob[cells]*wi*dp
            mass[cells] = mass[cells] - q
            energy[cells] = energy[cells] - h_up*q
        return mass, energy

    def step(self, state: GeothermalState, dt: float
             ) -> Tuple[GeothermalState, bool, int]:
        st0, st = state, state.copy()
        nc = self.g.cells.num
        m_scale = float((self.pv * 1000.0).sum()) / dt
        e_scale = float(np.abs(self.energy(st0.pressure, st0.T)).sum()) / dt

        for it in range(self.max_iterations):
            p, t = init_variables_adi(st.pressure, st.T)
            eqs = self.assemble(st0, p, t, dt)
            res = max(np.abs(value(eqs[0])).max() / m_scale,
                      np.abs(value(eqs[1])).max() / e_scale)
            if res <= self.tol:
                return st, True, it

            eq = combine_equations(*eqs)
            upd = spla.spsolve(eq.jac[0].tocsc(), -eq.val)
            st.pressure = st.pressure + upd[:nc]
            # limit the temperature step: the EOS is strongly nonlinear in T
            st.T = st.T + np.clip(upd[nc:], -25.0, 25.0)

        p, t = init_variables_adi(st.pressure, st.T)
        eqs = self.assemble(st0, p, t, dt)
        res = max(np.abs(value(eqs[0])).max() / m_scale,
                  np.abs(value(eqs[1])).max() / e_scale)
        return st, res <= self.tol, self.max_iterations


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.wells_and_bc import add_bc
    from pymrst.utils.gridtools import boundary_face_indices
    from pymrst.utils.units import BARSA, DARCY, DAY, MILLI, YEAR

    C = 273.15

    def _case(n=40, L=100.0, k=100 * MILLI * DARCY, poro=0.2):
        g = compute_geometry(cart_grid([n, 1, 1], [L, 10.0, 10.0]))
        nc = g.cells.num
        rock = make_rock(g, np.full((nc, 1), k), np.full((nc, 1), poro))
        return g, rock

    @dataclass
    class FrozenFluid(ThermalFluid):
        """Water with rho and mu held constant.

        Used where the temperature-property feedback would obscure what is
        being measured. The feedback itself is tested separately, in
        test_cold_water_is_harder_to_push.
        """

        def rho(self, p, t):
            return 1000.0 + 0.0 * value(t)

        def mu(self, p, t):
            return 1e-3 + 0.0 * value(t)

    def _sides(g, L):
        return (boundary_face_indices(g, "xmin"),
                boundary_face_indices(g, "xmax"))

    def test_steady_conduction_is_linear():
        """With no flow, conduction alone gives a linear temperature profile.

        Hold the two ends at different temperatures and equalise the
        pressures so nothing moves. The steady state of the heat equation
        with constant conductivity is a straight line - an exact answer to
        check the conductive term against.
        """
        L = 100.0
        g, rock = _case(n=40, L=L)
        fluid = ThermalFluid()
        left, right = _sides(g, L)

        p0 = 100 * BARSA
        bc = add_bc(None, left, "pressure", p0, sat=np.ones((left.size, 1)))
        bc = add_bc(bc, right, "pressure", p0, sat=np.ones((right.size, 1)))
        t_hot, t_cold = 150 + C, 50 + C
        bc_T = np.concatenate([np.full(left.size, t_hot),
                               np.full(right.size, t_cold)])

        model = GeothermalModel(g, rock, fluid, bc=bc, bc_T=bc_T)
        st = model.initial_state(p0, 50 + C)
        # Heat diffuses at ~7e-7 m2/s, so 100 m of rock needs centuries to
        # reach steady state - not the years a flow problem would take. Run
        # for 2000 years, well past L^2/alpha ~ 430 years.
        for _ in range(40):
            st, ok, _ = model.step(st, 50.0 * YEAR)
            assert ok

        x = g.cells.centroids[:, 0]
        want = t_hot + (t_cold - t_hot) * x / L
        err = np.abs(st.T - want).max()
        assert err < 0.5, f"steady conduction is not linear (max err {err:.2f} K)"
        # and nothing should have flowed
        assert np.abs(st.pressure - p0).max() / p0 < 1e-6
        print(f"ok test_steady_conduction_is_linear "
              f"(max deviation from the straight line {err:.3f} K)")

    def test_thermal_front_lags_the_water():
        """A cold front travels far slower than the water that carries it.

        The rock holds most of the heat, so the front is retarded by

            R = (pv rho Cw + (V - pv) rho_R Cp_R) / (pv rho Cw)

        This is the number that sets how long a geothermal doublet runs
        before the producer goes cold, so it is checked two ways: against the
        model's own energy function (exact), and against a traced front with
        conduction switched off (so advection alone sets the speed).
        """
        L = 100.0
        g, rock = _case(n=100, L=L, k=100 * MILLI * DARCY, poro=0.2)
        # Frozen properties and no conduction: the front is then pure
        # advection and its speed is exactly u / (phi R).
        fluid = FrozenFluid(lambda_f=0.0)
        no_conduction = ThermalRock(lambda_r=0.0)
        left, right = _sides(g, L)

        t_hot, t_cold = 150 + C, 40 + C
        p_in, p_out = 101 * BARSA, 100 * BARSA
        bc = add_bc(None, left, "pressure", p_in, sat=np.ones((left.size, 1)))
        bc = add_bc(bc, right, "pressure", p_out, sat=np.ones((right.size, 1)))
        bc_T = np.concatenate([np.full(left.size, t_cold),
                               np.full(right.size, t_hot)])

        model = GeothermalModel(g, rock, fluid, thermal_rock=no_conduction,
                                bc=bc, bc_T=bc_T)
        st = model.initial_state(p_out, t_hot)

        # what R does the model's own energy function imply? dE/dT is the
        # total heat capacity of a cell; the fluid's share of it is what the
        # water can carry. This is the sharpest test of the weighting.
        p_ad, t_ad = init_variables_adi(st.pressure.copy(), st.T.copy())
        dedt = np.diag(np.asarray(model.energy(p_ad, t_ad).jac[1].todense()))[0]
        dedt_fluid = model.pv[0] * 1000.0 * fluid.Cp
        R = dedt / dedt_fluid

        rho_f, vol, pv_cell = 1000.0, model.vol[0], model.pv[0]
        R_theory = ((pv_cell * rho_f * fluid.Cp
                     + (vol - pv_cell) * RHO_R * CP_R)
                    / (pv_cell * rho_f * fluid.Cp))
        assert np.isclose(R, R_theory, rtol=1e-12), (R, R_theory)
        assert 3.0 < R < 4.5, R

        # now trace the front and check it moves at u / (phi R)
        dt, nstep = 50.0 * DAY, 30
        volume_in = 0.0
        for k in range(nstep):
            st, ok, _ = model.step(st, dt)
            assert ok, f"step {k} failed"
            c = model.bc_cells[:left.size]
            mob = 1.0 / value(fluid.mu(st.pressure, st.T))[c]
            q = model.t_all[left] * mob * (p_in - st.pressure[c])
            volume_in += float(np.sum(q)) * dt

        x = g.cells.centroids[:, 0]
        mid = 0.5 * (t_hot + t_cold)
        assert st.T[0] < mid < st.T[-1], "the front has left the domain"
        front = float(np.interp(mid, st.T, x))

        pvi = volume_in / float(model.pv.sum())   # pore volumes injected
        x_water = pvi * L                         # one pore volume sweeps L
        ratio = x_water / front

        assert abs(ratio - R) / R < 0.1, (
            f"traced retardation {ratio:.2f} does not match the predicted "
            f"{R:.2f}")
        print(f"ok test_thermal_front_lags_the_water "
              f"(front at {front:.1f} m, water at {x_water:.1f} m "
              f"({pvi:.2f} pore volumes): retardation {ratio:.2f} vs the "
              f"predicted {R:.2f})")

    def test_conduction_runs_the_front_ahead_of_pure_advection():
        """With conduction on, the front leads its advective position.

        Not an error - the two transports add. Early on the diffusion length
        sqrt(alpha t) is comparable to the advective displacement, so the
        half-temperature point sits ahead of where advection alone would put
        it, and the apparent retardation only creeps up toward R as advection
        takes over. Worth pinning: it is why a retardation measured from a
        short simulation reads low.
        """
        L = 100.0
        g, rock = _case(n=100, L=L, k=100 * MILLI * DARCY, poro=0.2)
        left, right = _sides(g, L)
        t_hot, t_cold = 150 + C, 40 + C
        p_in, p_out = 101 * BARSA, 100 * BARSA

        def front_after(nstep, thermal_rock, lam_f):
            fluid = FrozenFluid(lambda_f=lam_f)
            bc = add_bc(None, left, "pressure", p_in,
                        sat=np.ones((left.size, 1)))
            bc = add_bc(bc, right, "pressure", p_out,
                        sat=np.ones((right.size, 1)))
            bc_T = np.concatenate([np.full(left.size, t_cold),
                                   np.full(right.size, t_hot)])
            model = GeothermalModel(g, rock, fluid, thermal_rock=thermal_rock,
                                    bc=bc, bc_T=bc_T)
            st = model.initial_state(p_out, t_hot)
            for _ in range(nstep):
                st, ok, _ = model.step(st, 50.0 * DAY)
                assert ok
            x = g.cells.centroids[:, 0]
            return float(np.interp(0.5 * (t_hot + t_cold), st.T, x))

        cold = front_after(22, ThermalRock(lambda_r=0.0), 0.0)    # advection only
        warm = front_after(22, ThermalRock(), LAMBDA_F)           # + conduction

        assert warm > cold, "conduction must carry the front further"
        assert warm < 2.0 * cold, "but it should not dominate advection here"
        print(f"ok test_conduction_runs_the_front_ahead_of_pure_advection "
              f"(front at {cold:.1f} m without conduction, {warm:.1f} m with)")

    def test_closed_system_conserves_energy_and_mass():
        """The energy equation conserves energy exactly in a sealed box.

        Held at constant fluid properties on purpose. A rigid, sealed,
        liquid-filled box is a thermo-mechanical trap: water's
        compressibility is ~4e-10 / Pa, so keeping the mass in a cell fixed
        while its temperature drops 1 K demands a ~28 bar pressure drop, and
        a 100 K swing drives the pressure negative. That is real physics
        (thermal pressurization), not a bug - but it swamps the thing being
        tested here, which is whether the energy equation conserves energy.
        Freezing rho and mu removes the coupling and leaves the conduction
        operator on its own.
        """
        # a small box, so conduction actually acts within the run
        g, rock = _case(n=20, L=2.0)
        fluid = FrozenFluid()
        model = GeothermalModel(g, rock, fluid)     # no bc: sealed and insulated

        st = model.initial_state(100 * BARSA, 60 + C)
        st.T[:10] = 200 + C

        e0 = float(model.energy(st.pressure, st.T).sum())
        spread0 = float(st.T.max() - st.T.min())

        for _ in range(30):
            st, ok, _ = model.step(st, 5.0 * DAY)
            assert ok

        e1 = float(model.energy(st.pressure, st.T).sum())
        spread1 = float(st.T.max() - st.T.min())

        assert abs(e1 - e0) / e0 < 1e-10, f"energy drifted by {(e1 - e0) / e0:.2e}"
        assert spread1 < 0.2 * spread0, "conduction should even the box out"
        # and it evens out toward the volume-weighted mean, not to anything else
        want = 0.5 * (200 + 60) + C
        assert abs(st.T.mean() - want) < 1.0, st.T.mean() - C
        print(f"ok test_closed_system_conserves_energy_and_mass "
              f"(spread {spread0:.0f} -> {spread1:.1f} K, energy conserved "
              f"to {abs(e1 - e0) / e0:.1e})")

    def test_thermal_pressurization_is_real():
        """Heating a sealed, liquid-filled box raises its pressure, hard.

        The coupling the test above deliberately removes. Worth pinning down:
        it is the reason a geothermal model cannot be run with a rigid sealed
        boundary and a realistic water EOS without the pressure running away.
        """
        g, rock = _case(n=5, L=5.0)
        model = GeothermalModel(g, rock, ThermalFluid())

        st = model.initial_state(100 * BARSA, 100 + C)
        p, t = init_variables_adi(st.pressure, st.T)
        rho = model.fluid.rho(p, t)
        # dRho/dp and dRho/dT at these conditions
        drho_dp = np.diag(np.asarray(rho.jac[0].todense()))[0]
        drho_dT = np.diag(np.asarray(rho.jac[1].todense()))[0]
        # holding the mass fixed, dp/dT = -(dRho/dT) / (dRho/dp)
        dp_dT = -drho_dT / drho_dp
        assert dp_dT > 5e5, f"expected a large thermal pressurization, got {dp_dT:.2e}"
        print(f"ok test_thermal_pressurization_is_real "
              f"(a sealed box gains {dp_dT / 1e5:.0f} bar per K of heating)")

    def test_cold_water_is_harder_to_push():
        """The EOS feeds back on the flow: cold water is more viscous.

        Injecting at fixed pressure drop, a cold reservoir takes less water
        than a hot one - purely because mu(T) changes. If the properties
        were frozen this test would fail.
        """
        L = 50.0
        g, rock = _case(n=20, L=L)
        fluid = ThermalFluid()
        left, right = _sides(g, L)

        def throughput(t_res):
            bc = add_bc(None, left, "pressure", 110 * BARSA,
                        sat=np.ones((left.size, 1)))
            bc = add_bc(bc, right, "pressure", 100 * BARSA,
                        sat=np.ones((right.size, 1)))
            bc_T = np.full(left.size + right.size, t_res)
            model = GeothermalModel(g, rock, fluid, bc=bc, bc_T=bc_T)
            st = model.initial_state(100 * BARSA, t_res)
            st, ok, _ = model.step(st, 1.0 * DAY)
            assert ok
            mu = float(fluid.mu(st.pressure, st.T).mean())
            return 1.0 / mu                        # mobility, up to constants

        cold = throughput(30 + C)
        hot = throughput(200 + C)
        assert hot > 4 * cold, (
            f"hot water should be far more mobile (hot {hot:.2e}, "
            f"cold {cold:.2e})")
        print(f"ok test_cold_water_is_harder_to_push "
              f"(mobility rises {hot / cold:.1f}x from 30 C to 200 C)")

    test_steady_conduction_is_linear()
    test_closed_system_conserves_energy_and_mass()
    test_thermal_pressurization_is_real()
    test_cold_water_is_harder_to_push()
    test_thermal_front_lags_the_water()
    test_conduction_runs_the_front_ahead_of_pure_advection()
    print("\nAll tests passed!")
