"""Shared coupled flow/mechanics kernels owned by the MRST-layout model tree.

Coupled flow and geomechanics (MRST ad-mechanics, fully coupled + splitting).

Poroelasticity changes exactly one thing about the flow equation: the pore
volume is no longer a function of pressure alone, but of the rock's
deformation too,

    effPV = V (phi pvMult(p) + alpha div(u))          (MRST equationsWaterMech)

and in the state-function graph that is *literally all it is* - a different
`PoreVolume`. Everything downstream (`ComponentPhaseMass`,
`ComponentTotalMass`, the conservation equations) is unchanged and does not
know that mechanics happened. This is what Phase 24 was for; the two models
below differ only in which `PoreVolume` they install.

- `MechWaterModel` (fully coupled): primary variables (p, xd), solved
  monolithically. `PoroMechanicalPoreVolume` reads div(u) from the
  mechanics, so the flow residual carries derivatives with respect to the
  displacement and the Jacobian is the true coupled one.

- `FixedStressSplitModel`: solve mechanics, then flow, then repeat.
  `FixedStressPoreVolume` substitutes the identity
  div(u) = sTerm + pTerm p, holding sTerm at the last mechanics solve but
  keeping pTerm p implicit. pTerm is the drained compressibility 1/K, so
  the flow solve sees the rock's stiffness even between mechanics solves -
  which is why fixed-stress converges where the naive (drained) split does
  not.

Both must converge to the same answer, and `test_fixed_stress_contracts_to_
fully_coupled` pins that.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse.linalg as spla

from pymrst.gridprocessing.grid import Grid
from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModelAD,
    blackoil_status,
)
from pymrst.modules.ad_core.models.facilities.facility_model import FacilityModel
from pymrst.modules.ad_core.models.reservoir_model import ReservoirModel
from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
from pymrst.modules.ad_core.statefunctions.pvt_property_functions import (
    simple_phase_fluid,
    water_phase_fluid,
)
from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    State,
    StateFunction,
)
from pymrst.modules.ad_mechanics.models.mechanic_model import MechanicModel
from pymrst.params.rock import Rock, pore_volume
from pymrst.solvers.trans import compute_trans
from pymrst.utils.adi import (
    ADI,
    combine_equations,
    init_variables_adi,
    value,
    vertcat_adi,
)


class PoroMechanicalPoreVolume(StateFunction):
    r"""$V(\phi\,\mathrm{pvMult}(p) + \alpha\,\mathrm{div}(u))$.

    The whole of poroelasticity, as far as the flow equation is concerned.
    """

    def __init__(self, model=None, scale_mech_by_volume: bool = True):
        super().__init__(model, label="\\Phi_{poromech}")
        self.depends_on(["pressure", "xd"], STATE)
        # False reproduces MRST's `equationsOilWaterMech`, which omits the
        # cell volume on the mechanical term - see the note in the module
        # docstring. Only for comparing against MRST; it is not correct.
        self.scale_mech_by_volume = scale_mech_by_volume

    def evaluate_on_domain(self, model, state):
        # Always via the *full* displacement: a stored (previous) state
        # carries its own `u`, so its div(u) is evaluated against the
        # boundary values that were in force when it was computed, not the
        # current ones. That matters the moment a displacement boundary
        # condition moves in time.
        u = (state["u"] if "u" in state
             else model.mech.full_displacement(state["xd"]))
        divu = model.mech.volumetric_strain_full(u)
        vol = model.G.cells.volumes
        mech = model.alpha * divu
        return (vol * model.rock.poro.ravel()
                * model.fluid.pv_mult(state["pressure"])
                + (vol * mech if self.scale_mech_by_volume else mech))


class FixedStressPoreVolume(StateFunction):
    r"""The same pore volume, with div(u) = sTerm + pTerm $\cdot$ p.

    An identity, not an approximation - but sTerm is held at the last
    mechanics solve while pTerm p stays implicit, which is the fixed-stress
    scheme.
    """

    def __init__(self, model=None, scale_mech_by_volume: bool = True):
        super().__init__(model, label="\\Phi_{fixed-stress}")
        self.depends_on("pressure", STATE)
        self.scale_mech_by_volume = scale_mech_by_volume

    def evaluate_on_domain(self, model, state):
        s_term, p_term = model.fixed_stress
        mech = model.alpha * (s_term + p_term * state["pressure"])
        vol = model.G.cells.volumes
        return (vol * model.rock.poro.ravel()
                * model.fluid.pv_mult(state["pressure"])
                + (vol * mech if self.scale_mech_by_volume else mech))


class _PoroFlowModel(ReservoirModel):
    """Single-phase flow whose pore volume is supplied by a subclass."""

    def __init__(self, g: Grid, rock: Rock, b, mu: float, rho: float,
                 alpha, bc_faces: np.ndarray, bc_values: np.ndarray,
                 pv_mult=None):
        fluid = water_phase_fluid(b, mu, rho)
        fluid.pv_mult = pv_mult or (lambda p: 1.0)
        operators = setup_operators_tpfa(g, rock)
        super().__init__(g, rock, fluid, operators, pore_volume(g, rock),
                         np.zeros(operators.T.size))
        self.alpha = np.broadcast_to(np.asarray(alpha, float).ravel(),
                                     (g.cells.num,)).copy()
        self.mu = mu

        # Dirichlet pressure faces: a half-transmissibility to the boundary.
        self.bc_faces = np.asarray(bc_faces, dtype=np.int64)
        self.bc_values = np.broadcast_to(
            np.asarray(bc_values, float).ravel(), self.bc_faces.shape).copy()
        t_half = compute_trans(g, rock)
        t_face = 1.0 / np.bincount(g.cells.faces, weights=1.0 / t_half,
                                   minlength=g.faces.num)
        self.bc_trans = t_face[self.bc_faces]
        self.bc_cells = g.faces.neighbors[self.bc_faces].sum(axis=1) + 1

    def boundary_flux(self, p):
        """Surface-volume outflow through the Dirichlet faces, per cell."""
        pc = p[self.bc_cells]
        dp = pc - self.bc_values
        b_face = self.fluid.b(p)[0][self.bc_cells]
        q = (self.bc_trans / self.mu) * b_face * dp
        out = np.zeros(self.G.cells.num)
        if isinstance(q, ADI):
            acc = ADI(out.copy(), [type(j)((self.G.cells.num, j.shape[1]))
                                   for j in q.jac])
            np.add.at(acc.val, self.bc_cells, q.val)
            import scipy.sparse as sp
            spread = sp.csr_array(
                (np.ones(self.bc_cells.size),
                 (self.bc_cells, np.arange(self.bc_cells.size))),
                shape=(self.G.cells.num, self.bc_cells.size),
            )
            acc.jac = [(spread @ j).tocsr() for j in q.jac]
            return acc
        np.add.at(out, self.bc_cells, q)
        return out

    def flow_equation(self, state0: State, state: State, dt: float):
        """The water conservation equation, straight out of the graph."""
        eq = self.conservation_equations(state0, state, dt)[0]
        return eq + self.boundary_flux(state["pressure"])


class MechWaterModel(_PoroFlowModel):
    """Fully coupled single-phase poroelasticity (MRST `MechWaterModel`)."""

    def __init__(self, g: Grid, rock: Rock, mech: MechanicModel, b, mu: float,
                 rho: float, bc_faces, bc_values, pv_mult=None,
                 tol: float = 1e-9, max_iterations: int = 20):
        super().__init__(g, rock, b, mu, rho, mech.alpha, bc_faces, bc_values,
                         pv_mult)
        self.mech = mech
        self.tol = tol
        self.max_iterations = max_iterations
        self.PVTPropertyFunctions.set_state_function(
            "PoreVolume", PoroMechanicalPoreVolume(self)
        )
        self.validate_model()

    def _finalize(self, st: State) -> State:
        """Store the full displacement, so this state's div(u) stays pinned to
        the boundary values it was solved with."""
        st["u"] = self.mech.full_displacement(st["xd"])
        return st

    def initial_state(self, p0: np.ndarray) -> State:
        """Undrained initial response: mechanics in equilibrium with p0."""
        p0 = np.asarray(p0, dtype=float)
        return self._finalize(State(pressure=p0, xd=self.mech.solve(p0)))

    def step(self, st0: State, dt: float) -> Tuple[State, bool, int]:
        """One backward-Euler step of the coupled system, by Newton."""
        st = st0.copy()
        st.pop("u", None)                 # rebuilt from xd as Newton proceeds
        nc = self.G.cells.num

        for it in range(self.max_iterations):
            p, xd = init_variables_adi(st["pressure"], st["xd"])
            cur = State(pressure=p, xd=xd)

            flow = self.flow_equation(st0, cur, dt)
            mech = self.mech.equations(xd, p)

            r_flow = float(np.max(np.abs(value(flow)))) / self.pv_ref.mean() * dt
            r_mech = float(np.max(np.abs(value(mech))))
            if max(r_flow, r_mech) < self.tol:
                return self._finalize(st), True, it

            eq = combine_equations(flow, mech)
            upd = spla.spsolve(eq.jac[0].tocsc(), -eq.val)
            st["pressure"] = st["pressure"] + upd[:nc]
            st["xd"] = st["xd"] + upd[nc:]
            st.clear_cache()

        return self._finalize(st), False, self.max_iterations


class MechWaterWellModel(ReservoirModel):
    """Well-driven single-phase poromechanics (MRST ``MechWaterModel``).

    This is the same water conservation equation as ``MechWaterModel``, but
    driven by the standard AD-core facility model instead of boundary faces.
    It is the path MRST's ``run2DCase('fluid_model', 'water')`` uses.
    """

    def __init__(self, g: Grid, rock: Rock, mech: MechanicModel, wells,
                 b, mu: float, rho: float, pv_mult=None,
                 # Deliberately *not* moved to MRST's 1e-3/1e-7 with the flow
                 # models. `tol_well` here also gates the mechanics residual
                 # through `max_extra_residual`, and that is a displacement
                 # balance in different units - not the normalized mass error
                 # CNV measures. Loosening it would trade an unmeasured amount
                 # of mechanics accuracy for flow iterations, so it stays until
                 # someone measures the coupled case.
                 tol_cnv: float = 1e-8, tol_mb: float = 1e-12,
                 tol_well: float = 1e-10, tol_mech: float = 1e-9):
        fluid = water_phase_fluid(b, mu, rho)
        fluid.pv_mult = pv_mult or (lambda p: 1.0)
        operators = setup_operators_tpfa(g, rock)
        super().__init__(g, rock, fluid, operators, pore_volume(g, rock),
                         np.zeros(operators.T.size))
        self.mech = mech
        self.alpha = mech.alpha
        self.facility = FacilityModel(wells)
        self.tol_cnv = tol_cnv
        self.tol_mb = tol_mb
        self.tol_well = tol_well
        self.tol_mech = tol_mech
        self.PVTPropertyFunctions.set_state_function(
            "PoreVolume", PoroMechanicalPoreVolume(self)
        )
        self.validate_model()

    @property
    def wells(self):
        return self.facility.wells

    def _q_names(self) -> List[str]:
        return ["qWs"]

    def init_state_ad(self, st: State) -> State:
        p, qws, bhp, xd = init_variables_adi(
            st["pressure"], st["qWs"], st["bhp"], st["xd"]
        )
        return State(pressure=p, qWs=qws, bhp=bhp, xd=xd)

    def _state0_props(self, st0: State) -> State:
        out = State(pressure=st0["pressure"], xd=st0["xd"], u=st0["u"])
        if "well_flux" in st0:
            out["well_flux"] = st0["well_flux"]
        return out

    def initial_state(self, p0: np.ndarray) -> State:
        p0 = np.asarray(p0, dtype=float)
        nw = self.facility.n_wells
        qws = np.zeros(nw)
        bhp = np.zeros(nw)
        for k, w in enumerate(self.wells):
            if w.type == "rate":
                qws[k] = w.val
                bhp[k] = float(np.mean(p0))
            else:
                bhp[k] = w.val
        st = State(pressure=p0, qWs=qws, bhp=bhp,
                   xd=self.mech.solve(p0))
        st["u"] = self.mech.full_displacement(st["xd"])
        return st

    def get_equations(self, state0: State, state: State, dt: float):
        st = self.init_state_ad(state)
        self._conv_state = st
        st0 = self._state0_props(state0)

        eqs = self.conservation_equations(st0, st, dt)
        cq_s, well_eqs, ctrl = self.facility.well_equations(
            self, st, [st["qWs"]], st["bhp"], state0=st0
        )
        eqs = self.facility.apply_to_reservoir(eqs, cq_s)
        mech = self.mech.equations(st["xd"], st["pressure"])
        return eqs + well_eqs + [ctrl, mech]

    def get_convergence(self, eqs, dt: float) -> Tuple[bool, Dict[str, float]]:
        cell_eq = np.asarray(value(eqs[0]), dtype=float)
        well_eq = np.asarray(value(eqs[1]), dtype=float)
        ctrl = np.asarray(value(eqs[2]), dtype=float)
        mech = np.asarray(value(eqs[3]), dtype=float)

        st = getattr(self, "_conv_state", None)
        if st is None:
            b_avg = 1.0
            pv = self.pv_ref
        else:
            b = self.get_props(st, "ShrinkageFactors")[0]
            b_avg = float(np.mean(1.0 / np.asarray(value(b), dtype=float)))
            pv = np.asarray(value(self.get_props(st, "PoreVolume")), dtype=float)

        cnv = b_avg * np.abs(cell_eq / pv).max() * dt
        mb = b_avg * abs(cell_eq.sum()) * dt / pv.sum()
        rates = [abs(w.val) for w in self.wells if w.type != "bhp"]
        q_ref = max(rates) if rates else 1.0
        well = float(np.max(np.abs(well_eq)) / q_ref) if well_eq.size else 0.0
        ctrl_scale = np.array([
            1e5 if w.type == "bhp" else max(abs(w.val), 1e-12)
            for w in self.wells
        ])
        ctrl_res = (float(np.max(np.abs(ctrl) / ctrl_scale))
                    if ctrl.size else 0.0)
        mech_res = float(np.max(np.abs(mech))) if mech.size else 0.0
        converged = (cnv <= self.tol_cnv and mb <= self.tol_mb
                     and well <= self.tol_well
                     and ctrl_res <= self.tol_well
                     and mech_res <= self.tol_mech
                     and self.facility.within_limits)
        return converged, {
            "CNV": cnv, "MB": mb, "well": well, "control": ctrl_res,
            "mech": mech_res,
        }

    def update_state(self, st: State, increment: np.ndarray) -> State:
        nc, nw = self.G.cells.num, self.facility.n_wells
        n_free = self.mech.n_free
        new = st.copy()
        new["pressure"] = st["pressure"] + increment[:nc]
        off = nc
        new["qWs"] = st["qWs"] + increment[off:off + nw]
        off += nw
        new["bhp"] = st["bhp"] + increment[off:off + nw]
        off += nw
        new["xd"] = st["xd"] + increment[off:off + n_free]
        new["u"] = self.mech.full_displacement(new["xd"])
        self.facility.update_switched_controls(new, self._q_names())
        new.clear_cache()
        return new

    def after_convergence(self, state: State, state0: State,
                          dt: float) -> State:
        if self.facility.last_flux is not None:
            state["well_flux"] = [q.copy() for q in self.facility.last_flux]
        return state


class FixedStressWaterWellModel(ReservoirModel):
    """Well-driven fixed-stress single-phase poromechanics.

    This mirrors MRST's ``MechFluidFixedStressSplitModel`` with
    ``fluidModelType='water'``: mechanics is solved at the current pressure,
    the fixed-stress ``sTerm + pTerm*p`` identity is installed in the flow
    pore-volume term, and the well-driven water equation is solved.
    """

    def __init__(self, g: Grid, rock: Rock, mech: MechanicModel, wells,
                 b, mu: float, rho: float, pv_mult=None,
                 tol: float = 1e-8, splitting_tol: float = 1e-6,
                 tol_well: float = 1e-8, max_iterations: int = 20,
                 max_splitting_iterations: int = 100):
        fluid = water_phase_fluid(b, mu, rho)
        fluid.pv_mult = pv_mult or (lambda p: 1.0)
        operators = setup_operators_tpfa(g, rock)
        super().__init__(g, rock, fluid, operators, pore_volume(g, rock),
                         np.zeros(operators.T.size))
        self.mech = mech
        self.alpha = mech.alpha
        self.facility = FacilityModel(wells)
        self.tol = tol
        self.splitting_tol = splitting_tol
        self.tol_well = tol_well
        self.max_iterations = max_iterations
        self.max_splitting_iterations = max_splitting_iterations
        self.fixed_stress = (np.zeros(g.cells.num), np.zeros(g.cells.num))
        self.fixed_stress_old = (np.zeros(g.cells.num), np.zeros(g.cells.num))
        self.PVTPropertyFunctions.set_state_function(
            "PoreVolume", FixedStressPoreVolume(self)
        )
        self.validate_model()

    @property
    def wells(self):
        return self.facility.wells

    def _q_names(self) -> List[str]:
        return ["qWs"]

    def initial_state(self, p0: np.ndarray) -> State:
        p0 = np.asarray(p0, dtype=float)
        nw = self.facility.n_wells
        qws = np.zeros(nw)
        bhp = np.zeros(nw)
        for k, w in enumerate(self.wells):
            if w.type == "rate":
                qws[k] = w.val
                bhp[k] = float(np.mean(p0))
            else:
                bhp[k] = w.val
        xd = self.mech.solve(p0)
        return State(pressure=p0, qWs=qws, bhp=bhp, xd=xd,
                     u=self.mech.full_displacement(xd))

    def _flow_state_ad(self, st: State) -> State:
        p, qws, bhp = init_variables_adi(st["pressure"], st["qWs"], st["bhp"])
        return State(pressure=p, qWs=qws, bhp=bhp)

    def _previous_flow_state(self, st0: State) -> State:
        out = State(pressure=st0["pressure"])
        if "well_flux" in st0:
            out["well_flux"] = st0["well_flux"]
        return out

    def _flow_equations(self, st0: State, st: State, dt: float):
        cur = self._flow_state_ad(st)
        prev = self._previous_flow_state(st0)

        saved = self.fixed_stress
        self.fixed_stress = self.fixed_stress_old
        pv0 = self.get_props(prev, "PoreVolume")
        self.fixed_stress = saved

        pv = self.get_props(cur, "PoreVolume")
        b = self.get_props(cur, "ShrinkageFactors")[0]
        b0 = self.fluid.b(prev["pressure"])[0]
        flux = self.get_props(cur, "ComponentTotalFlux")[0]
        eq = [(pv * b - pv0 * value(b0)) / dt + self.operators.div(flux)]

        cq_s, well_eqs, ctrl = self.facility.well_equations(
            self, cur, [cur["qWs"]], cur["bhp"], state0=prev
        )
        eq = self.facility.apply_to_reservoir(eq, cq_s)
        return eq + well_eqs + [ctrl]

    def _flow_convergence(self, eqs, st: State, dt: float):
        cell_eq = np.asarray(value(eqs[0]), dtype=float)
        well_eq = np.asarray(value(eqs[1]), dtype=float)
        ctrl = np.asarray(value(eqs[2]), dtype=float)
        cur = State(pressure=st["pressure"])
        b = self.fluid.b(st["pressure"])[0]
        b_avg = float(np.mean(1.0 / np.asarray(value(b), dtype=float)))
        pv = np.asarray(value(self.get_props(cur, "PoreVolume")), dtype=float)
        cnv = b_avg * np.abs(cell_eq / pv).max() * dt
        mb = b_avg * abs(cell_eq.sum()) * dt / pv.sum()
        rates = [abs(w.val) for w in self.wells if w.type != "bhp"]
        q_ref = max(rates) if rates else 1.0
        well = float(np.max(np.abs(well_eq)) / q_ref) if well_eq.size else 0.0
        ctrl_scale = np.array([
            1e5 if w.type == "bhp" else max(abs(w.val), 1e-12)
            for w in self.wells
        ])
        ctrl_res = (float(np.max(np.abs(ctrl) / ctrl_scale))
                    if ctrl.size else 0.0)
        ok = (cnv <= self.tol and mb <= self.tol
              and well <= self.tol_well and ctrl_res <= self.tol_well
              and self.facility.within_limits)
        return ok, {"CNV": cnv, "MB": mb, "well": well, "control": ctrl_res}

    def _update_flow_state(self, st: State, increment: np.ndarray) -> State:
        nc, nw = self.G.cells.num, self.facility.n_wells
        new = st.copy()
        new["pressure"] = st["pressure"] + increment[:nc]
        off = nc
        new["qWs"] = st["qWs"] + increment[off:off + nw]
        off += nw
        new["bhp"] = st["bhp"] + increment[off:off + nw]
        self.facility.update_switched_controls(new, self._q_names())
        new.clear_cache()
        return new

    def _solve_flow(self, st0: State, st_start: State,
                    dt: float) -> Tuple[State, bool, Dict[str, Any]]:
        st = st_start.copy()
        residual_history: List[Dict[str, float]] = []
        line_searches = 0
        for it in range(self.max_iterations):
            eqs = self._flow_equations(st0, st, dt)
            ok, residuals = self._flow_convergence(eqs, st, dt)
            residual_history.append(dict(residuals))
            if ok:
                return st, True, {"iterations": it,
                                  "residuals": residual_history,
                                  "line_searches": line_searches,
                                  "converged": True}
            eq = combine_equations(*eqs)
            upd = spla.spsolve(eq.jac[0].tocsc(), -eq.val)
            # The field-scale oil-water well equations can overshoot badly
            # from the initially oil-filled state.  Accept the first Newton
            # fraction that lowers the scaled CNV/MB/well/control norm.  This
            # is the split-model counterpart of NonLinearSolver's globalization.
            norm0 = max(residuals.values())
            best_state = None
            best_norm = np.inf
            alpha = 1.0
            for search in range(6):
                candidate = self._update_flow_state(st, alpha * upd)
                candidate_eqs = self._flow_equations(st0, candidate, dt)
                _, candidate_residuals = self._flow_convergence(
                    candidate_eqs, candidate, dt
                )
                candidate_norm = max(candidate_residuals.values())
                if candidate_norm < best_norm:
                    best_state, best_norm = candidate, candidate_norm
                if candidate_norm <= norm0:
                    line_searches += search
                    break
                alpha *= 0.5
            else:
                line_searches += 6
            st = best_state
        eqs = self._flow_equations(st0, st, dt)
        ok, residuals = self._flow_convergence(eqs, st, dt)
        residual_history.append(dict(residuals))
        return st, ok, {"iterations": self.max_iterations,
                        "residuals": residual_history,
                        "line_searches": line_searches,
                        "converged": bool(ok)}

    def step(self, st0: State, dt: float) -> Tuple[State, bool, int]:
        st = st0.copy()
        outer_history: List[Dict[str, Any]] = []
        self.last_split_report = {
            "converged": False, "iterations": 0, "outer": outer_history,
            "failure": "",
        }
        self.fixed_stress_old = self.mech.fixed_stress_terms(
            st0["xd"], st0["pressure"]
        )

        for it in range(self.max_splitting_iterations):
            xd = self.mech.solve(st["pressure"])
            st["xd"] = xd
            st["u"] = self.mech.full_displacement(xd)
            self.fixed_stress = self.mech.fixed_stress_terms(
                xd, st["pressure"]
            )
            st, ok, flow_report = self._solve_flow(st0, st, dt)
            if not ok:
                outer_history.append({"iteration": it + 1,
                                      "mechanics_residual": None,
                                      "flow": flow_report})
                self.last_split_report.update(
                    iterations=it + 1,
                    failure="water Newton solve did not converge",
                )
                return st, False, it

            res = float(np.max(np.abs(
                self.mech.equations(xd, st["pressure"])
            )))
            outer_history.append({"iteration": it + 1,
                                  "mechanics_residual": res,
                                  "flow": flow_report})
            if res < self.splitting_tol:
                xd = self.mech.solve(st["pressure"])
                st["xd"] = xd
                st["u"] = self.mech.full_displacement(xd)
                if self.facility.last_flux is not None:
                    st["well_flux"] = [
                        q.copy() for q in self.facility.last_flux
                    ]
                self.last_split_report.update(
                    converged=True, iterations=it + 1, failure="",
                )
                return st, True, it + 1

        self.last_split_report.update(
            iterations=self.max_splitting_iterations,
            failure="water fixed-stress iteration did not converge",
        )
        return st, False, self.max_splitting_iterations


class FixedStressOilWaterWellModel(ReservoirModel):
    """Well-driven fixed-stress oil-water poromechanics.

    Matches MRST's ``OilWaterFixedStressFluidModel`` path. The flow residual
    is assembled explicitly so the previous accumulation uses the old
    fixed-stress mechanics terms while the current accumulation uses the new
    terms, exactly as MRST passes them through ``drivingForces``.
    """

    def __init__(self, g: Grid, rock: Rock, fluid, wells,
                 mech: MechanicModel, pv_mult=None,
                 tol: float = 1e-8, splitting_tol: float = 1e-6,
                 tol_well: float = 1e-8, max_iterations: int = 30,
                 max_splitting_iterations: int = 100,
                 mrst_oilwater_bug: bool = True,
                 ds_max: float = 0.2, gravity: float = 0.0):
        phase_fluid = simple_phase_fluid(fluid)
        phase_fluid.pv_mult = pv_mult or phase_fluid.pv_mult
        operators = setup_operators_tpfa(g, rock)
        if gravity and g.griddim >= 3:
            z = g.cells.centroids[:, 2]
            gdz = gravity * (z[operators.N[:, 1]] - z[operators.N[:, 0]])
        else:
            gdz = np.zeros(operators.T.size)
        super().__init__(g, rock, phase_fluid, operators,
                         pore_volume(g, rock), gdz, gravity=gravity)
        self.raw_fluid = fluid
        self.mech = mech
        self.alpha = mech.alpha
        self.facility = FacilityModel(wells)
        self.tol = tol
        self.splitting_tol = splitting_tol
        self.tol_well = tol_well
        self.max_iterations = max_iterations
        self.max_splitting_iterations = max_splitting_iterations
        self.scale_mech_by_volume = not mrst_oilwater_bug
        self.ds_max = ds_max
        self.fixed_stress = (np.zeros(g.cells.num), np.zeros(g.cells.num))
        self.fixed_stress_old = (np.zeros(g.cells.num), np.zeros(g.cells.num))
        # Keep a valid state function for diagnostics/get_props calls. The
        # actual fixed-stress flow residual below computes old/new PV directly.
        self.PVTPropertyFunctions.set_state_function(
            "PoreVolume",
            FixedStressPoreVolume(
                self, scale_mech_by_volume=self.scale_mech_by_volume
            ),
        )
        self.validate_model()

    @property
    def wells(self):
        return self.facility.wells

    def _q_names(self) -> List[str]:
        return ["qWs", "qOs"]

    def initial_state(self, p0: np.ndarray, sw0: float = 0.0) -> State:
        p0 = np.asarray(p0, dtype=float)
        nw = self.facility.n_wells
        qws = np.zeros(nw)
        qos = np.zeros(nw)
        bhp = np.zeros(nw)
        for k, w in enumerate(self.wells):
            if w.type == "rate":
                qws[k] = w.val * w.compi[0]
                qos[k] = w.val * w.compi[1]
                bhp[k] = float(np.mean(p0))
            else:
                bhp[k] = w.val
        xd = self.mech.solve(p0)
        return State(pressure=p0, sW=np.full(self.G.cells.num, sw0),
                     sO=np.full(self.G.cells.num, 1.0 - sw0),
                     qWs=qws, qOs=qos, bhp=bhp, xd=xd,
                     u=self.mech.full_displacement(xd))

    def _flow_state_ad(self, st: State) -> State:
        p, sw, qws, qos, bhp = init_variables_adi(
            st["pressure"], st["sW"], st["qWs"], st["qOs"], st["bhp"]
        )
        return State(pressure=p, sW=sw, qWs=qws, qOs=qos, bhp=bhp)

    def _previous_flow_state(self, st0: State) -> State:
        out = State(pressure=st0["pressure"], sW=st0["sW"])
        if "well_flux" in st0:
            out["well_flux"] = st0["well_flux"]
        return out

    def _fixed_pv(self, p, terms):
        s_term, p_term = terms
        vol = self.G.cells.volumes
        mech = self.alpha * (s_term + p_term * p)
        return (vol * self.rock.poro.ravel() * self.fluid.pv_mult(p)
                + (vol * mech if self.scale_mech_by_volume else mech))

    def _flow_equations(self, st0: State, st: State, dt: float):
        cur = self._flow_state_ad(st)
        prev = self._previous_flow_state(st0)

        pv = self._fixed_pv(cur["pressure"], self.fixed_stress)
        pv0 = self._fixed_pv(prev["pressure"], self.fixed_stress_old)
        bw, bo = self.get_props(cur, "ShrinkageFactors")
        bw0, bo0 = self.fluid.b(prev["pressure"])
        flux_w, flux_o = self.get_props(cur, "ComponentTotalFlux")
        sw, sw0 = cur["sW"], prev["sW"]
        eqs = [
            (pv * bw * sw - pv0 * value(bw0) * sw0) / dt
            + self.operators.div(flux_w),
            (pv * bo * (1.0 - sw) - pv0 * value(bo0) * (1.0 - sw0)) / dt
            + self.operators.div(flux_o),
        ]

        cq_s, well_eqs, ctrl = self.facility.well_equations(
            self, cur, [cur["qWs"], cur["qOs"]], cur["bhp"], state0=prev
        )
        eqs = self.facility.apply_to_reservoir(eqs, cq_s)
        return eqs + well_eqs + [ctrl]

    def _flow_convergence(self, eqs, st: State, dt: float):
        water, oil, eq_w, eq_o, ctrl = [np.asarray(value(e), dtype=float)
                                        for e in eqs]
        bw, bo = self.fluid.b(st["pressure"])
        b_avg = np.array([
            float(np.mean(1.0 / np.asarray(value(bw), dtype=float))),
            float(np.mean(1.0 / np.asarray(value(bo), dtype=float))),
        ])
        pv = np.asarray(value(self._fixed_pv(
            st["pressure"], self.fixed_stress
        )), dtype=float)
        cnv = max(b_avg[0] * np.abs(water / pv).max(),
                  b_avg[1] * np.abs(oil / pv).max()) * dt
        mb = max(b_avg[0] * abs(water.sum()),
                 b_avg[1] * abs(oil.sum())) * dt / pv.sum()
        rates = [abs(w.val) for w in self.wells if w.type != "bhp"]
        q_ref = max(rates) if rates else 1.0
        well = max(float(np.max(np.abs(eq_w))) if eq_w.size else 0.0,
                   float(np.max(np.abs(eq_o))) if eq_o.size else 0.0) / q_ref
        ctrl_scale = np.array([
            1e5 if w.type == "bhp" else max(abs(w.val), 1e-12)
            for w in self.wells
        ])
        ctrl_res = (float(np.max(np.abs(ctrl) / ctrl_scale))
                    if ctrl.size else 0.0)
        ok = (cnv <= self.tol and mb <= self.tol
              and well <= self.tol_well and ctrl_res <= self.tol_well
              and self.facility.within_limits)
        return ok, {"CNV": cnv, "MB": mb, "well": well, "control": ctrl_res}

    def _update_flow_state(self, st: State, increment: np.ndarray) -> State:
        nc, nw = self.G.cells.num, self.facility.n_wells
        new = st.copy()
        new["pressure"] = st["pressure"] + increment[:nc]
        dsw = np.clip(increment[nc:2 * nc], -self.ds_max, self.ds_max)
        new["sW"] = np.clip(st["sW"] + dsw, 0.0, 1.0)
        new["sO"] = 1.0 - new["sW"]
        off = 2 * nc
        new["qWs"] = st["qWs"] + increment[off:off + nw]
        off += nw
        new["qOs"] = st["qOs"] + increment[off:off + nw]
        off += nw
        new["bhp"] = st["bhp"] + increment[off:off + nw]
        self.facility.update_switched_controls(new, self._q_names())
        new.clear_cache()
        return new

    def _solve_flow(self, st0: State, st_start: State,
                    dt: float) -> Tuple[State, bool, Dict[str, Any]]:
        st = st_start.copy()
        residual_history: List[Dict[str, float]] = []
        for it in range(self.max_iterations):
            eqs = self._flow_equations(st0, st, dt)
            ok, residuals = self._flow_convergence(eqs, st, dt)
            residual_history.append(dict(residuals))
            if ok:
                return st, True, {"iterations": it,
                                  "residuals": residual_history,
                                  "converged": True}
            eq = combine_equations(*eqs)
            upd = spla.spsolve(eq.jac[0].tocsc(), -eq.val)
            st = self._update_flow_state(st, upd)
        eqs = self._flow_equations(st0, st, dt)
        ok, residuals = self._flow_convergence(eqs, st, dt)
        residual_history.append(dict(residuals))
        return st, ok, {"iterations": self.max_iterations,
                        "residuals": residual_history,
                        "converged": bool(ok)}

    def step(self, st0: State, dt: float) -> Tuple[State, bool, int]:
        st = st0.copy()
        outer_history: List[Dict[str, Any]] = []
        self.last_split_report = {
            "converged": False, "iterations": 0, "outer": outer_history,
            "failure": "",
        }
        self.fixed_stress_old = self.mech.fixed_stress_terms(
            st0["xd"], st0["pressure"]
        )

        for it in range(self.max_splitting_iterations):
            xd = self.mech.solve(st["pressure"])
            st["xd"] = xd
            st["u"] = self.mech.full_displacement(xd)
            self.fixed_stress = self.mech.fixed_stress_terms(
                xd, st["pressure"]
            )
            st, ok, flow_report = self._solve_flow(st0, st, dt)
            if not ok:
                outer_history.append({"iteration": it + 1,
                                      "mechanics_residual": None,
                                      "flow": flow_report})
                self.last_split_report.update(
                    iterations=it + 1,
                    failure="oil-water Newton solve did not converge",
                )
                return st, False, it

            res = float(np.max(np.abs(
                self.mech.equations(xd, st["pressure"])
            )))
            outer_history.append({"iteration": it + 1,
                                  "mechanics_residual": res,
                                  "flow": flow_report})
            if res < self.splitting_tol:
                xd = self.mech.solve(st["pressure"])
                st["xd"] = xd
                st["u"] = self.mech.full_displacement(xd)
                if self.facility.last_flux is not None:
                    st["well_flux"] = [
                        q.copy() for q in self.facility.last_flux
                    ]
                self.last_split_report.update(
                    converged=True, iterations=it + 1, failure="",
                )
                return st, True, it + 1

        self.last_split_report.update(
            iterations=self.max_splitting_iterations,
            failure="oil-water fixed-stress iteration did not converge",
        )
        return st, False, self.max_splitting_iterations


class FixedStressBlackOilWellModel(GenericBlackOilModelAD):
    """Well-driven fixed-stress black-oil poromechanics.

    This mirrors MRST's ``BlackOilFixedStressFluidModel`` path: mechanics is
    solved at the current pressure, the flow solve uses the fixed-stress
    identity for pore volume, and the black-oil mass/flux/well machinery stays
    the standard dissolved-gas model.
    """

    def __init__(self, g: Grid, rock: Rock, fluid, wells,
                 mech: MechanicModel, tol: float = 1e-8,
                 splitting_tol: float = 1e-6, tol_well: float = 1e-8,
                 max_iterations: int = 30,
                 max_splitting_iterations: int = 100,
                 ds_max: float = 0.2, gravity: float = 0.0):
        super().__init__(
            g, rock, fluid, wells, disgas=True, gravity=gravity,
            tol_cnv=tol, tol_mb=tol, tol_well=tol_well, ds_max=ds_max,
        )
        self.mech = mech
        self.alpha = mech.alpha
        self.splitting_tol = splitting_tol
        self.max_iterations = max_iterations
        self.max_splitting_iterations = max_splitting_iterations
        self.fixed_stress = (np.zeros(g.cells.num), np.zeros(g.cells.num))
        self.fixed_stress_old = (np.zeros(g.cells.num), np.zeros(g.cells.num))
        self.PVTPropertyFunctions.set_state_function(
            "PoreVolume", FixedStressPoreVolume(self)
        )
        self.validate_model()

    def _q_names(self) -> List[str]:
        return ["qWs", "qOs", "qGs"][:self.n_phases]

    def initial_state(self, p0: np.ndarray, sw0: float = 0.0,
                      sg0: float = 0.0, rs=None) -> State:
        p0 = np.asarray(p0, dtype=float)
        sw = np.full(self.G.cells.num, sw0, dtype=float)
        sg = np.full(self.G.cells.num, sg0, dtype=float)
        so = 1.0 - sw - sg
        if rs is None:
            rs = 0.5 * self.raw_fluid.rs_sat(p0)
        rs = np.asarray(rs, dtype=float)
        nw = self.facility.n_wells
        qws = np.zeros(nw)
        qos = np.zeros(nw)
        qgs = np.zeros(nw)
        bhp = np.zeros(nw)
        for k, w in enumerate(self.wells):
            if w.type == "rate":
                qws[k] = w.val * w.compi[0]
                qos[k] = w.val * w.compi[1]
                qgs[k] = w.val * w.compi[2]
                bhp[k] = float(np.mean(p0))
            else:
                bhp[k] = w.val
        xd = self.mech.solve(p0)
        return State(
            pressure=p0, sW=sw, sO=so, sG=sg, rs=rs,
            status=blackoil_status(sw, so, sg, disgas=True),
            qWs=qws, qOs=qos, qGs=qgs, bhp=bhp, xd=xd,
            u=self.mech.full_displacement(xd),
        )

    def _component_total_mass_with_terms(self, st: State, terms):
        saved = self.fixed_stress
        self.fixed_stress = terms
        try:
            return self.get_props(st.copy(), "ComponentTotalMass")
        finally:
            self.fixed_stress = saved

    def _flow_equations(self, st0: State, st: State, dt: float):
        cur = self.init_state_ad(st)
        self._conv_state = cur
        prev = self._state0_props(st0)

        mass = self._component_total_mass_with_terms(cur, self.fixed_stress)
        mass0 = self._component_total_mass_with_terms(
            prev, self.fixed_stress_old
        )
        flux = self.get_props(cur, "ComponentTotalFlux")
        eqs = [
            (mass[i] - mass0[i]) / dt + self.operators.div(flux[i])
            for i in range(self.n_components)
        ]

        q_s = [cur[q] for q in self._q_names()]
        cq_s, well_eqs, ctrl = self.facility.well_equations(
            self, cur, q_s, cur["bhp"], state0=prev
        )
        eqs = self.facility.apply_to_reservoir(
            eqs, cq_s, self._basis_scale()
        )
        return eqs + well_eqs + [ctrl]

    def _solve_flow(self, st0: State, st_start: State,
                    dt: float) -> Tuple[State, bool, Dict[str, Any]]:
        st = st_start.copy()
        residual_history: List[Dict[str, float]] = []
        for it in range(self.max_iterations):
            eqs = self._flow_equations(st0, st, dt)
            ok, residuals = self.get_convergence(eqs, dt)
            residual_history.append(dict(residuals))
            if ok:
                return st, True, {"iterations": it,
                                  "residuals": residual_history,
                                  "converged": True}
            eq = combine_equations(*eqs)
            upd = spla.spsolve(eq.jac[0].tocsc(), -eq.val)
            # Live-oil well controls can alternate between two otherwise
            # converged Newton states (most visibly for a gas-rate injector
            # opposite a BHP producer).  Backtrack on the complete black-oil
            # convergence norm, matching the globalization used by MRST's
            # nonlinear solver instead of accepting that control cycle.
            norm0 = max(residuals.values())
            best = None
            best_norm = np.inf
            alpha = 1.0
            for _ in range(6):
                candidate = self.update_state(st, alpha * upd)
                candidate_eqs = self._flow_equations(st0, candidate, dt)
                _, candidate_residuals = self.get_convergence(
                    candidate_eqs, dt
                )
                candidate_norm = max(candidate_residuals.values())
                if candidate_norm < best_norm:
                    best, best_norm = candidate, candidate_norm
                if candidate_norm <= norm0:
                    break
                alpha *= 0.5
            st = best
        eqs = self._flow_equations(st0, st, dt)
        ok, residuals = self.get_convergence(eqs, dt)
        residual_history.append(dict(residuals))
        return st, ok, {"iterations": self.max_iterations,
                        "residuals": residual_history,
                        "converged": bool(ok)}

    def step(self, st0: State, dt: float) -> Tuple[State, bool, int]:
        st = st0.copy()
        outer_history: List[Dict[str, Any]] = []
        self.last_split_report = {
            "converged": False, "iterations": 0, "outer": outer_history,
            "failure": "",
        }
        self.fixed_stress_old = self.mech.fixed_stress_terms(
            st0["xd"], st0["pressure"]
        )

        for it in range(self.max_splitting_iterations):
            xd = self.mech.solve(st["pressure"])
            st["xd"] = xd
            st["u"] = self.mech.full_displacement(xd)
            self.fixed_stress = self.mech.fixed_stress_terms(
                xd, st["pressure"]
            )
            st, ok, flow_report = self._solve_flow(st0, st, dt)
            if not ok:
                outer_history.append({"iteration": it + 1,
                                      "mechanics_residual": None,
                                      "flow": flow_report})
                self.last_split_report.update(
                    iterations=it + 1,
                    failure="black-oil Newton solve did not converge",
                )
                return st, False, it

            res = float(np.max(np.abs(
                self.mech.equations(xd, st["pressure"])
            )))
            outer_history.append({"iteration": it + 1,
                                  "mechanics_residual": res,
                                  "flow": flow_report})
            if res < self.splitting_tol:
                xd = self.mech.solve(st["pressure"])
                st["xd"] = xd
                st["u"] = self.mech.full_displacement(xd)
                st = self.after_convergence(st, st0, dt)
                self.last_split_report.update(
                    converged=True, iterations=it + 1, failure="",
                )
                return st, True, it + 1

        self.last_split_report.update(
            iterations=self.max_splitting_iterations,
            failure="black-oil fixed-stress iteration did not converge",
        )
        return st, False, self.max_splitting_iterations


class FixedStressSplitModel(_PoroFlowModel):
    """Fixed-stress sequential splitting (MRST `MechFluidFixedStressSplitModel`).

    Each outer iteration: solve mechanics at the current pressure, recompute
    the fixed-stress terms, then solve flow with them. Converged when the
    mechanics residual at the new pressure is small.
    """

    def __init__(self, g: Grid, rock: Rock, mech: MechanicModel, b, mu: float,
                 rho: float, bc_faces, bc_values, pv_mult=None,
                 tol: float = 1e-9, splitting_tol: float = 1e-10,
                 max_splitting_iterations: int = 100,
                 max_iterations: int = 20):
        super().__init__(g, rock, b, mu, rho, mech.alpha, bc_faces, bc_values,
                         pv_mult)
        self.mech = mech
        self.tol = tol
        self.splitting_tol = splitting_tol
        self.max_splitting_iterations = max_splitting_iterations
        self.max_iterations = max_iterations
        self.fixed_stress = (np.zeros(g.cells.num), np.zeros(g.cells.num))
        self.fixed_stress_old = (np.zeros(g.cells.num), np.zeros(g.cells.num))
        self.PVTPropertyFunctions.set_state_function(
            "PoreVolume", FixedStressPoreVolume(self)
        )
        self.validate_model()

    def initial_state(self, p0: np.ndarray) -> State:
        p0 = np.asarray(p0, dtype=float)
        return State(pressure=p0, xd=self.mech.solve(p0))

    def _solve_flow(self, st0: State, p_start: np.ndarray,
                    dt: float) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """Flow alone, with the fixed-stress pore volume."""
        p_cur = p_start.copy()
        history: List[float] = []
        # The previous step's pore volume uses the previous mechanics state.
        self.fixed_stress, self.fixed_stress_old = (
            self.fixed_stress, self.fixed_stress_old
        )
        for iteration in range(self.max_iterations):
            (p,) = init_variables_adi(p_cur)
            cur = State(pressure=p)

            saved = self.fixed_stress
            self.fixed_stress = self.fixed_stress_old
            prev = State(pressure=st0["pressure"])
            pv0 = self.get_props(prev, "PoreVolume")
            self.fixed_stress = saved

            eq = self._flow_with_pv0(prev, cur, pv0, dt)
            res = float(np.max(np.abs(value(eq)))) / self.pv_ref.mean() * dt
            history.append(res)
            if res < self.tol:
                return p_cur, True, {
                    "iterations": iteration,
                    "residuals": history,
                    "converged": True,
                }
            upd = spla.spsolve(eq.jac[0].tocsc(), -eq.val)
            p_cur = p_cur + upd
        return p_cur, False, {
            "iterations": self.max_iterations,
            "residuals": history,
            "converged": False,
        }

    def _flow_with_pv0(self, prev: State, cur: State, pv0, dt: float):
        """Water conservation with the previous pore volume supplied."""
        pv = self.get_props(cur, "PoreVolume")
        b = self.get_props(cur, "ShrinkageFactors")[0]
        b0 = self.fluid.b(prev["pressure"])[0]
        flux = self.get_props(cur, "ComponentTotalFlux")[0]
        acc = (pv * b - pv0 * value(b0)) / dt
        return acc + self.operators.div(flux) \
            + self.boundary_flux(cur["pressure"])

    def step(self, st0: State, dt: float) -> Tuple[State, bool, int]:
        """One step by fixed-stress iteration between mechanics and flow."""
        p = st0["pressure"].copy()
        xd = st0["xd"].copy()
        outer_history: List[Dict[str, Any]] = []
        self.last_split_report = {
            "converged": False, "iterations": 0, "outer": outer_history,
            "failure": "",
        }

        self.fixed_stress_old = self.mech.fixed_stress_terms(
            st0["xd"], st0["pressure"]
        )

        for it in range(self.max_splitting_iterations):
            xd = self.mech.solve(p)                       # 1. mechanics
            self.fixed_stress = self.mech.fixed_stress_terms(xd, p)
            p, ok, flow_report = self._solve_flow(st0, p, dt)  # 2. flow
            if not ok:
                outer_history.append({
                    "iteration": it + 1,
                    "mechanics_residual": None,
                    "flow": flow_report,
                })
                self.last_split_report.update(
                    iterations=it + 1,
                    failure="fluid Newton solve did not converge",
                )
                return State(pressure=p, xd=xd), False, it

            # 3. converged when mechanics still balances the new pressure
            res = float(np.max(np.abs(self.mech.equations(xd, p))))
            outer_history.append({
                "iteration": it + 1,
                "mechanics_residual": res,
                "flow": flow_report,
            })
            if res < self.splitting_tol:
                xd = self.mech.solve(p)
                self.last_split_report.update(
                    converged=True, iterations=it + 1, failure="",
                )
                return State(pressure=p, xd=xd), True, it + 1

        self.last_split_report.update(
            iterations=self.max_splitting_iterations,
            failure="fixed-stress iteration did not converge",
        )
        return State(pressure=p, xd=xd), False, self.max_splitting_iterations


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.vemmech.vem import (
        ElasticBC,
        create_augmented_grid,
        enu2c,
    )
    from pymrst.params.rock import make_rock
    from pymrst.utils.units import DARCY

    def terzaghi_analytic(z_over_L, tau, nterms=200):
        """p/p0 for consolidation drained at z=0, sealed at z=L."""
        out = np.zeros_like(z_over_L)
        for m in range(nterms):
            n = 2 * m + 1
            out += (4.0 / (np.pi * n)) * np.sin(n * np.pi * z_over_L / 2.0) \
                * np.exp(-(n * np.pi / 2.0) ** 2 * tau)
        return out

    def build_terzaghi(nz=30, L=10.0):
        """The classic column: loaded and drained at the top, sealed below."""
        g = create_augmented_grid(
            compute_geometry(cart_grid([1, 1, nz], [1.0, 1.0, L]))
        )
        E, nu, alpha = 1e9, 0.25, 1.0
        perm, mu_f, sigma0 = 1.0 * DARCY, 1e-3, 1e6
        c_f = 1e-9                      # slightly compressible water

        C = enu2c(np.full(nz, E), np.full(nz, nu))
        rock = make_rock(g, perm, 0.2)

        pts, fc = g.nodes.coords, g.faces.centroids
        bd = np.any(g.faces.neighbors == -1, axis=1)
        top_f = np.flatnonzero(bd & (fc[:, 2] < 1e-9))
        bot = np.flatnonzero(pts[:, 2] > L - 1e-9)
        nx = np.flatnonzero((pts[:, 0] < 1e-9) | (pts[:, 0] > 1 - 1e-9))
        ny = np.flatnonzero((pts[:, 1] < 1e-9) | (pts[:, 1] > 1 - 1e-9))
        dn = np.concatenate([bot, nx, ny])
        dm = np.vstack([
            np.tile([False, False, True], (bot.size, 1)),
            np.tile([True, False, False], (nx.size, 1)),
            np.tile([False, True, False], (ny.size, 1)),
        ])
        el_bc = ElasticBC(
            disp_nodes=dn, disp_mask=dm, disp_vals=np.zeros((dn.size, 3)),
            force_faces=top_f,
            force=np.tile([0.0, 0.0, sigma0], (top_f.size, 1)),
        )
        mech = MechanicModel(g, C, alpha=alpha, el_bc=el_bc)

        def b(p):
            return 1.0 + c_f * p

        kv = E * (1 - nu) / ((1 + nu) * (1 - 2 * nu))
        return dict(g=g, rock=rock, mech=mech, b=b, mu=mu_f, rho=1000.0,
                    bc_faces=top_f, bc_values=np.zeros(top_f.size),
                    kv=kv, sigma0=sigma0, perm=perm, L=L, alpha=alpha,
                    poro=0.2, c_f=c_f)

    def test_terzaghi_fully_coupled():
        """1D consolidation through the coupled model matches the analytic
        series - the same benchmark the linear BiotModel passes, but now
        through the state-function graph and a nonlinear (compressible) fluid.
        """
        cfg = build_terzaghi()
        m = MechWaterModel(cfg["g"], cfg["rock"], cfg["mech"], cfg["b"],
                           cfg["mu"], cfg["rho"], cfg["bc_faces"],
                           cfg["bc_values"])
        kv, sigma0, L = cfg["kv"], cfg["sigma0"], cfg["L"]

        # Storage: 1/M = phi c_f (the rock grains are incompressible, alpha=1)
        one_over_M = cfg["poro"] * cfg["c_f"]
        p0_val = sigma0 / (kv * one_over_M + 1.0)
        cv = (cfg["perm"] / cfg["mu"]) / (one_over_M + 1.0 / kv)

        st = m.initial_state(np.full(cfg["g"].cells.num, p0_val))
        z = cfg["g"].cells.centroids[:, 2] / L

        for tau_target, tau_prev in ((0.1, 0.0), (0.5, 0.1)):
            t = (tau_target - tau_prev) * L**2 / cv
            for _ in range(60):
                st, ok, _ = m.step(st, t / 60)
                assert ok, "coupled Newton failed"
            exact = p0_val * terzaghi_analytic(z, tau_target)
            err = np.max(np.abs(st["pressure"] - exact)) / p0_val
            assert err < 0.03, f"tau={tau_target}: rel err {err:.4f}"
            print(f"  tau={tau_target}: max rel err {err:.4f}")
        print("ok test_terzaghi_fully_coupled")

    def test_fixed_stress_contracts_to_fully_coupled():
        """THE splitting test: the sequential scheme must converge to the
        monolithic answer, not merely to something stable."""
        cfg = build_terzaghi(nz=12)
        kv, sigma0 = cfg["kv"], cfg["sigma0"]
        one_over_M = cfg["poro"] * cfg["c_f"]
        p0_val = sigma0 / (kv * one_over_M + 1.0)
        cv = (cfg["perm"] / cfg["mu"]) / (one_over_M + 1.0 / kv)
        dt = 0.05 * cfg["L"] ** 2 / cv

        full = MechWaterModel(cfg["g"], cfg["rock"], cfg["mech"], cfg["b"],
                              cfg["mu"], cfg["rho"], cfg["bc_faces"],
                              cfg["bc_values"])
        split = FixedStressSplitModel(cfg["g"], cfg["rock"], cfg["mech"],
                                      cfg["b"], cfg["mu"], cfg["rho"],
                                      cfg["bc_faces"], cfg["bc_values"])

        p_init = np.full(cfg["g"].cells.num, p0_val)
        st_f = full.initial_state(p_init)
        st_s = split.initial_state(p_init)

        worst = 0.0
        for _ in range(5):
            st_f, ok_f, _ = full.step(st_f, dt)
            st_s, ok_s, its = split.step(st_s, dt)
            assert ok_f and ok_s, "a step failed to converge"
            rel = np.max(np.abs(st_s["pressure"] - st_f["pressure"])) / p0_val
            worst = max(worst, rel)
        assert worst < 1e-6, f"splitting did not contract: {worst:.2e}"
        print(f"ok test_fixed_stress_contracts_to_fully_coupled "
              f"(max rel diff {worst:.1e}, {its} outer its/step)")

    def test_pore_volume_is_the_only_difference():
        """The coupling really is one state function: the poromechanical
        pore volume differs from the flow-only one by exactly alpha*V*div(u).

        The sign follows the effective stress, so the column compacts when
        the applied load dominates, dilates when the pressure does, and does
        neither when they balance (alpha = 1, so they cancel at p = sigma0).
        """
        cfg = build_terzaghi(nz=8)
        m = MechWaterModel(cfg["g"], cfg["rock"], cfg["mech"], cfg["b"],
                           cfg["mu"], cfg["rho"], cfg["bc_faces"],
                           cfg["bc_values"])
        vol = cfg["g"].cells.volumes
        flow_only = vol * cfg["rock"].poro.ravel()
        sigma0, kv = cfg["sigma0"], cfg["kv"]

        for p_val, expect in ((0.0, "compact"), (sigma0, "balance"),
                              (2 * sigma0, "dilate")):
            st = m.initial_state(np.full(cfg["g"].cells.num, p_val))
            pv = m.get_props(st, "PoreVolume")
            divu = m.mech.volumetric_strain(st["xd"])

            # the identity that makes this a drop-in PoreVolume
            assert np.allclose(pv - flow_only, cfg["alpha"] * vol * divu,
                               rtol=1e-10)
            # and the closed form for uniaxial strain: div(u) = (p - sigma0)/kv
            assert np.allclose(divu, (p_val - sigma0) / kv, rtol=1e-6), expect
            if expect == "compact":
                assert np.all(pv < flow_only)
            elif expect == "dilate":
                assert np.all(pv > flow_only)
        print("ok test_pore_volume_is_the_only_difference")

    test_pore_volume_is_the_only_difference()
    test_terzaghi_fully_coupled()
    test_fixed_stress_contracts_to_fully_coupled()
    print("\nAll tests passed!")
