"""Sequential pressure/transport splitting (MRST sequential).

The fully-implicit system couples pressure and saturation in one Newton
solve. Splitting it exploits the fact that pressure is nearly elliptic and
saturation nearly hyperbolic:

1. **Pressure step** - sum the phase equations (weighted by the surface
   densities) so the saturation-derivative terms cancel to leading order,
   and solve the resulting near-elliptic equation for p with saturations
   frozen. This yields the total velocity vT.
2. **Transport step** - hold vT fixed and solve the saturation equation,
   with each phase flux written in fractional-flow form

       q_ph = f_ph * (vT + sum_other T * mob_other * (g_ph - g_other))

   (computeSequentialFluxes). Gravity enters through the second term.

Optionally the two are iterated to convergence ("outer" iterations), which
recovers the fully-implicit solution exactly; a single pass is the classic
IMPES-like sequential scheme, whose error is a splitting error of O(dt).
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.modules.ad_blackoil.models.two_phase_oil_water_model import (
    OWState,
    TwoPhaseOilWaterModelAD,
)
from pymrst.utils.adi import (
    ad_sum,
    combine_equations,
    init_variables_adi,
    value,
    vertcat_adi,
)


@dataclass
class SequentialReport:
    """Iteration counts of one sequential step."""

    pressure_its: int
    transport_its: int
    outer_its: int
    converged: bool


class SequentialOilWaterModel:
    """Sequential pressure/transport solver for the AD oil-water model.

    Wraps a validated fully-implicit model and reuses its residual
    assembly, so the two solve *the same* discrete equations - the only
    difference is how the Newton system is split.
    """

    def __init__(self, fi: TwoPhaseOilWaterModelAD,
                 outer_tol: float = 1e-6, max_outer: int = 1,
                 max_its: int = 25):
        """max_outer = 1 gives the classic single-pass (IMPES-like) scheme;
        larger values iterate pressure and transport until the fully-implicit
        residual is met. Outer iterations converge *linearly* (the splitting
        error contracts by a factor set by the compressibility coupling), so
        recovering the fully-implicit answer takes tens of passes."""
        self.fi = fi
        self.outer_tol = outer_tol
        self.max_outer = max_outer
        self.max_its = max_its

    # -- pressure: solve for p with sW frozen -----------------------------
    def _pressure_step(self, st0: OWState, st: OWState, dt: float
                       ) -> Tuple[OWState, int]:
        """Solve the summed (near-elliptic) equation for p; sW is frozen, so
        it simply is not declared as an AD variable and the system is square
        in (p, qWs, qOs, bhp)."""
        fi = self.fi
        nc, nw = fi.g.cells.num, len(fi.wells)
        st = st.copy()
        tol = fi.tol_cnv * float(np.abs(fi.pv).max()) / dt

        for it in range(self.max_its):
            p, qws, qos, bhp = init_variables_adi(
                st.pressure, st.qWs, st.qOs, st.bhp
            )
            water, oil, eq_w, eq_o, ctrl = fi.assemble(
                st0, p, st.sW, qws, qos, bhp, dt
            )
            eqs = [water + oil, eq_w, eq_o, ctrl]
            res = np.concatenate([np.atleast_1d(value(e)) for e in eqs])
            if np.abs(res).max() <= tol:
                return st, it
            eq = combine_equations(*eqs)
            upd = spla.spsolve(eq.jac[0].tocsc(), -eq.val)
            cap = fi.dp_max_rel * np.abs(st.pressure)
            st.pressure = st.pressure + np.clip(upd[:nc], -cap, cap)
            st.qWs = st.qWs + upd[nc:nc + nw]
            st.qOs = st.qOs + upd[nc + nw:nc + 2 * nw]
            st.bhp = st.bhp + upd[nc + 2 * nw:nc + 3 * nw]
        return st, self.max_its

    # -- transport: solve for sW with p and the total velocity frozen -----
    def _transport_step(self, st0: OWState, st: OWState, dt: float
                        ) -> Tuple[OWState, int]:
        """Solve the water (saturation) equation with the pressure field
        frozen. The well unknowns stay in the system so the well rates are
        consistent with the new saturations."""
        fi = self.fi
        nc, nw = fi.g.cells.num, len(fi.wells)
        st = st.copy()
        tol = fi.tol_cnv * float(np.abs(fi.pv).max()) / dt

        for it in range(self.max_its):
            sw, qws, qos, bhp = init_variables_adi(
                st.sW, st.qWs, st.qOs, st.bhp
            )
            water, _, eq_w, eq_o, ctrl = fi.assemble(
                st0, st.pressure, sw, qws, qos, bhp, dt
            )
            eqs = [water, eq_w, eq_o, ctrl]
            res = np.concatenate([np.atleast_1d(value(e)) for e in eqs])
            if np.abs(res).max() <= tol:
                return st, it
            eq = combine_equations(*eqs)
            upd = spla.spsolve(eq.jac[0].tocsc(), -eq.val)
            st.sW = np.clip(
                st.sW + np.clip(upd[:nc], -fi.ds_max, fi.ds_max), 0.0, 1.0
            )
            st.qWs = st.qWs + upd[nc:nc + nw]
            st.qOs = st.qOs + upd[nc + nw:nc + 2 * nw]
            st.bhp = st.bhp + upd[nc + 2 * nw:nc + 3 * nw]
        return st, self.max_its

    def step(self, state: OWState, dt: float
             ) -> Tuple[OWState, SequentialReport]:
        """One sequential step: pressure, then transport (optionally outer-
        iterated until the fully-implicit residual is satisfied)."""
        fi = self.fi
        st0 = state
        st = state.copy()
        p_its = t_its = 0
        converged = False
        for outer in range(self.max_outer):
            st, pi = self._pressure_step(st0, st, dt)
            st, ti = self._transport_step(st0, st, dt)
            p_its += pi
            t_its += ti
            # check the *fully-implicit* residual: if it is satisfied, the
            # split solution is the fully-implicit solution.
            p, sw, qws, qos, bhp = init_variables_adi(
                st.pressure, st.sW, st.qWs, st.qOs, st.bhp
            )
            eqs = fi.assemble(st0, p, sw, qws, qos, bhp, dt)
            if fi._converged(eqs, dt, st):
                converged = True
                break
        return st, SequentialReport(p_its, t_its, outer + 1, converged)


if __name__ == "__main__":
    import time

    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.ad_props import init_simple_adi_fluid
    from pymrst.params.rock import make_rock
    from pymrst.params.wells_and_bc import add_well
    from pymrst.utils.units import BARSA, CENTI, DARCY, DAY, MILLI, POISE

    def _model(nx=15):
        g = compute_geometry(cart_grid([nx, nx, 1], [300.0, 300.0, 10.0]))
        kvals = (50 + (11 * np.arange(g.cells.num)) % 200) * MILLI * DARCY
        rock = make_rock(g, kvals[:, None], 0.25)
        fluid = init_simple_adi_fluid(
            mu=np.array([1.0, 5.0]) * CENTI * POISE, rho=[1000.0, 800.0],
            n=[2, 2], c=[1e-6 / BARSA, 1e-4 / BARSA], p_ref=200 * BARSA,
        )
        w = add_well(None, g, rock, np.array([0]), well_type="rate",
                     val=2e-3, compi=np.array([1.0, 0.0]), sign=1)
        w = add_well(w, g, rock, np.array([g.cells.num - 1]),
                     well_type="bhp", val=150 * BARSA,
                     compi=np.array([0.0, 1.0]), sign=-1)
        return TwoPhaseOilWaterModelAD(g, rock, fluid, w)

    def test_outer_iterations_contract_toward_fully_implicit():
        """Each outer pass shrinks the fully-implicit residual: the splitting
        error contracts (linearly), so more passes give an answer closer to
        the fully-implicit one."""
        fi = _model()
        st0 = fi.initial_state(200 * BARSA, 0.2)
        dt = 30 * DAY

        st_fi, ok, _ = fi.step(st0, dt)
        assert ok

        errs = []
        for n_outer in (1, 5, 20):
            seq = SequentialOilWaterModel(fi, max_outer=n_outer)
            st, _ = seq.step(st0, dt)
            errs.append(float(np.abs(st.sW - st_fi.sW).max()))

        assert errs[0] > errs[1] > errs[2], errs
        assert errs[2] < 0.05 * errs[0], errs
        print(f"ok test_outer_iterations_contract_toward_fully_implicit "
              f"(max |dsW|: {errs[0]:.2e} -> {errs[1]:.2e} -> {errs[2]:.2e})")

    def test_single_pass_splitting_error_shrinks_with_dt():
        """One pressure + one transport pass per step (the classic sequential
        scheme) has a splitting error that vanishes as the timestep does."""
        fi = _model()
        seq = SequentialOilWaterModel(fi, max_outer=1)
        total = 30.0 * DAY

        errs = []
        for nsteps in (1, 4, 16):
            dt = total / nsteps
            st_fi = fi.initial_state(200 * BARSA, 0.2)
            st_sq = fi.initial_state(200 * BARSA, 0.2)
            for _ in range(nsteps):
                st_fi, ok, _ = fi.step(st_fi, dt)
                assert ok
                st_sq, _ = seq.step(st_sq, dt)
            errs.append(float(np.abs(st_sq.sW - st_fi.sW).max()))

        # The error decays with dt, but the constant is large near a sharp
        # front: a single pass freezes the mobilities used by the pressure
        # solve, which is exactly where the displacement front is moving.
        assert errs[0] > errs[1] > errs[2], errs
        assert errs[2] < 0.4 * errs[0], errs
        print(f"ok test_single_pass_splitting_error_shrinks_with_dt "
              f"(max |dsW| = {errs[0]:.2f} -> {errs[1]:.2f} -> {errs[2]:.2f} "
              f"as dt drops 16x)")

    def test_water_mass_is_conserved():
        """The transport step solves the water equation exactly, so injected
        water is conserved to solver tolerance even in a single pass."""
        fi = _model()
        seq = SequentialOilWaterModel(fi, max_outer=1)
        st = fi.initial_state(200 * BARSA, 0.2)
        f, pv = fi.fluid, fi.pv
        w0 = float(np.sum(pv * value(f.b_w(st.pressure)) * st.sW))
        dt = 10 * DAY
        states = []
        for _ in range(4):
            st, _ = seq.step(st, dt)
            states.append(st)
        w1 = float(np.sum(pv * value(f.b_w(st.pressure)) * st.sW))
        injected = dt * sum(s.qWs.sum() for s in states)
        assert np.isclose(w1 - w0, injected, rtol=1e-5), (w1 - w0, injected)
        print("ok test_water_mass_is_conserved")

    test_outer_iterations_contract_toward_fully_implicit()
    test_single_pass_splitting_error_shrinks_with_dt()
    test_water_mass_is_conserved()
    print("\nAll tests passed!")
