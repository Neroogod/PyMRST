"""MRST ``ad-core/models/ReservoirModel.m``.

The reservoir layer of MRST's AD-OO framework: rock, fluid and phases on top of
`PhysicalModel`, the three state-function groupings that turn a state into
properties, the facility (wells) and aquifer submodels, and the update rules
that keep a Newton increment physical.

State follows MRST's layout. Saturations are one `s` matrix with a column per
active phase in water, oil, gas order (a list of columns while it carries
derivatives); wells live in `wellSol`, one row per well. `model.fluid` is the
MRST fluid itself - `bW`, `muO`, `krW`, `rhoGS`, ... - not PyMRST's legacy
`PhaseFluid`.

Indices are 0-based. `get_phase_indices` returns `-1` for an inactive phase
where MRST also returns `-1`, and the position (`0`, `1`, `2`) where MRST
returns a one-based one.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from pymrst.modules.ad_core.ad_oo.physical_model import (
    WHOLE,
    PhysicalModel,
    StepReport,
    merge_options_relaxed,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.params.gravity.gravity import gravity
from pymrst.utils.adi import ad_matmul, value

PHASE_LETTERS: str = "WOG"
LONG_PHASE_NAMES: tuple = ("water", "oil", "gas")
SATURATION_WARNING_TOLERANCE: float = 1e-8


class ReservoirModel(PhysicalModel):
    """MRST `ReservoirModel`: a flow model on a grid with rock and fluid."""

    #: MRST tests `isa(model, 'GenericReservoirModel')` here, to pick the
    #: generic facility and to treat the equations as mass rather than surface
    #: volume. `GenericReservoirModel` sets this; reading a flag avoids
    #: importing the subclass into its own base.
    isGenericReservoirModel = False

    def __init__(self, G: Any, rock: Any = None, fluid: Any = None,
                 **options: Any):
        super().__init__(G)
        self.rock = rock
        # MRST's fluid is a struct read by field; a fluid given as a dict
        # (`initDeckADIFluid` in this port builds one) becomes a record with
        # the same fields, so `fluid.bO` and `isfield(fluid, 'pcOW')` both
        # read the way MRST's code does.
        self.fluid = SimpleNamespace(**fluid) if isinstance(fluid, dict) else fluid
        do_setup = G is not None and rock is not None
        self.dpMaxRel = float("inf")
        self.dpMaxAbs = float("inf")
        self.minimumPressure = -float("inf")
        self.maximumPressure = float("inf")
        self.dsMaxAbs = 0.2
        self.nonlinearTolerance = 1e-6
        self.inputdata = None
        self.useCNVConvergence = False
        self.toleranceCNV = 1e-3
        self.toleranceMB = 1e-7
        self.extraStateOutput = False
        self.extraWellSolOutput = True
        self.outputFluxes = True
        self.gravity = gravity()
        self.FacilityModel = self._default_facility_model()
        self.FacilityNonLinearSolver = None
        self.AquiferModel = None
        self.FlowPropertyFunctions = None
        self.PVTPropertyFunctions = None
        self.FlowDiscretization = None
        self.Components: List[Any] = []
        self.water = False
        self.gas = False
        self.oil = False
        merge_options_relaxed(self, options)
        self.water = False
        self.gas = False
        self.oil = False
        if do_setup:
            self.setup_operators()

    def _default_facility_model(self) -> Any:
        if self.isGenericReservoirModel:
            from pymrst.modules.ad_core.ad_oo.generic_facility_model import (
                GenericFacilityModel,
            )
            return GenericFacilityModel()
        from pymrst.modules.ad_core.ad_oo.facility_model import FacilityModel
        return FacilityModel()

    # -- validation ---------------------------------------------------------

    def validate_state(self, state: State) -> State:
        state = super().validate_state(state)
        active = self.get_active_phases()
        n_ph = int(np.count_nonzero(active))
        nc = self.G.cells.num
        self.check_property(state, "Pressure", [nc, 1], [1, 2])
        if n_ph > 1:
            self.check_property(state, "Saturation", [nc, n_ph], [1, 2])
        if "s" in state:
            total = np.sum(_saturation_matrix(state["s"]), axis=1)
            off_unity = np.abs(total - 1) > SATURATION_WARNING_TOLERANCE
            if np.any(off_unity):
                import warnings
                warnings.warn(f"Saturations deviate from unity in "
                              f"{int(off_unity.sum())} cells")
        if self.FacilityModel is not None:
            state = self.FacilityModel.validate_state(state)
        if self.AquiferModel is not None:
            state = self.AquiferModel.validate_state(state)
        if "sMax" not in state and "s" in state:
            state["sMax"] = _saturation_matrix(state["s"]).copy()
        return state

    def get_saturation_var_names(self) -> List[str]:
        names = ["sw", "so", "sg"]
        return [n for n, on in zip(names, self.get_active_phases()) if on]

    def get_maximum_timestep(self, state: State, state0: State, dt0: float,
                             driving_forces: Any) -> float:
        dt = super().get_maximum_timestep(state, state0, dt0, driving_forces)
        if self.FlowDiscretization is not None:
            dt_f = self.FlowDiscretization.get_maximum_timestep(
                self, state, state0, dt0, driving_forces)
            dt = min(dt, dt_f)
        return dt

    def prepare_report_step(self, state: State, state0: State, dt: float,
                            driving_forces: Any):
        if _wells_of(driving_forces):
            assert self.FacilityModel is not None, (
                "FacilityModel not set up. Call model.validateModel before "
                "calling equations with wells.")
            self.FacilityModel, state = self.FacilityModel.prepare_report_step(
                state, state0, dt, driving_forces)
        return self, state

    def prepare_timestep(self, state: State, state0: State, dt: float,
                         driving_forces: Any):
        if self.FacilityModel is not None and _wells_of(driving_forces):
            self.FacilityModel, state = self.FacilityModel.prepare_timestep(
                state, state0, dt, driving_forces)
            self.FacilityModel = self.FacilityModel.set_reservoir_model(self)
        if self.FlowDiscretization is not None:
            self.FlowDiscretization, state = self.FlowDiscretization.prepare_timestep(
                self, state, state0, dt, driving_forces)
        return self, state

    def validate_model(self, *args: Any) -> "ReservoirModel":
        model = super().validate_model(*args)
        if model.FacilityModel is None:
            from pymrst.modules.ad_core.ad_oo.facility_model import FacilityModel
            model.FacilityModel = FacilityModel(model)
        assert model.operators is not None, (
            "Operators must be set up before simulation. See "
            "model.setupOperators for more details.")
        if getattr(model.operators, "gdz", None) is None:
            model.operators.gdz = model.get_gravity_gradient()
        if model.AquiferModel is not None:
            model.AquiferModel = model.AquiferModel.validate_model(model)
        model.FacilityModel = model.FacilityModel.set_reservoir_model(model)
        model.FacilityModel = model.FacilityModel.validate_model(*args)
        if model.inputdata is not None:
            model.minimumPressure = 0.0
        return model

    def validate_driving_forces(self, ctrl: Any, index: int) -> Any:
        """Normalise producer `compi` and refuse an injector's bad one."""
        nph = self.get_number_of_phases()
        for i, w in enumerate(_wells_of(ctrl)):
            is_prod = w.sign < 0
            if nph == 1:
                w.compi = np.array([1.0])
                continue
            compi = (np.zeros(0) if w.compi is None
                     else np.atleast_1d(np.asarray(w.compi, dtype=float)))
            if compi.size != nph:
                if compi.size == 0 and is_prod:
                    if self.verbose:
                        print(f"Well {i + 1}, control {index + 1}: Defaulting "
                              f"producer compi field.")
                    w.compi = np.full(nph, 1.0/nph)
                    continue
                raise ValueError(
                    f"Well .compi field must contain 1 by {nph} entries for a "
                    f"{nph}-phase model.\nControl {index + 1}, well {w.name} "
                    f"(#{i + 1}) provided 1 by {compi.size} entries.")
            if abs(compi.sum() - 1) > 1e-8:
                if not is_prod:
                    raise ValueError(
                        f"Well {w.name} (#{i + 1}) for control {index + 1} has "
                        f".compi that does not sum to one.")
                total = compi.sum()
                w.compi = compi/total if total > 0 else np.full(nph, 1.0/nph)
        return ctrl

    # -- state-function groupings -------------------------------------------

    def setup_state_function_groupings(self, *args: Any) -> "ReservoirModel":
        from pymrst.modules.ad_core.ad_oo.statefunctions.flow_discretization import (
            FlowDiscretization,
        )
        from pymrst.modules.ad_core.ad_oo.statefunctions.flow_property_functions import (
            FlowPropertyFunctions,
        )
        from pymrst.modules.ad_core.ad_oo.statefunctions.pvt_property_functions import (
            PVTPropertyFunctions,
        )
        if self.FlowPropertyFunctions is None:
            self.FlowPropertyFunctions = FlowPropertyFunctions(self)
        if self.PVTPropertyFunctions is None:
            self.PVTPropertyFunctions = PVTPropertyFunctions(self)
        if self.FlowDiscretization is None:
            self.FlowDiscretization = FlowDiscretization(self)
        if self.FacilityModel is not None:
            self.FacilityModel = self.FacilityModel.setup_state_function_groupings(*args)
        return self

    def set_state_function_evaluation_mode(self, mode: str) -> "ReservoirModel":
        super().set_state_function_evaluation_mode(mode)
        if self.FacilityModel is not None:
            self.FacilityModel = self.FacilityModel.set_state_function_evaluation_mode(mode)
        return self

    def remove_state_function_groupings(self) -> "ReservoirModel":
        self.FlowPropertyFunctions = None
        self.PVTPropertyFunctions = None
        self.FlowDiscretization = None
        if self.FacilityModel is not None:
            self.FacilityModel = self.FacilityModel.remove_state_function_groupings()
        return self

    def get_state_function_groupings(self):
        groups, names, models = super().get_state_function_groupings()
        extra_names = ["FlowPropertyFunctions", "PVTPropertyFunctions",
                       "FlowDiscretization"]
        extra_groups = [getattr(self, n) for n in extra_names]
        extra_models = [self]*len(extra_names)
        if self.FacilityModel is not None:
            fm_groups, fm_names, fm_models = \
                self.FacilityModel.get_state_function_groupings()
            extra_groups += list(fm_groups)
            extra_names += list(fm_names)
            extra_models += list(fm_models)
        for group, name, model in zip(extra_groups, extra_names, extra_models):
            if group is not None:
                groups.append(group)
                names.append(name)
                models.append(model)
        return groups, names, models

    # -- the Newton step ----------------------------------------------------

    def step_function(self, state: State, state0: State, dt: float,
                      driving_forces: Any, linsolver: Any, nonlinsolver: Any,
                      iteration: int, **options: Any):
        facility_report = None
        if iteration == 1 and self.FacilityNonLinearSolver is not None:
            state, facility_report = self.FacilityNonLinearSolver.solve_timestep(
                state0, dt, self.FacilityModel, initialGuess=state,
                W=_wells_of(driving_forces))
        state, report = super().step_function(state, state0, dt, driving_forces,
                                              linsolver, nonlinsolver,
                                              iteration, **options)
        report["FacilityStepReport"] = facility_report
        return state, report

    def update_for_changed_controls(self, state: State, forces: Any):
        from pymrst.modules.ad_core.utils.init_well_sol_ad import init_well_sol_ad
        self.FacilityModel = self.FacilityModel.setup_wells(_wells_of(forces))
        state["wellSol"] = init_well_sol_ad(_wells_of(forces), self, state)
        return super().update_for_changed_controls(state, forces)

    def update_state(self, state: State, problem: Any, dx: Sequence[Any],
                     driving_forces: Any):
        rest_vars, sat_vars, _ = self.split_primary_variables(
            problem.primary_variables)
        if "wellSol" in state:
            state, _ = self.FacilityModel.update_state(state, problem, dx,
                                                       driving_forces)
        state = self.update_saturations(state, dx, problem, sat_vars)
        if rest_vars:
            state, _, _ = self.update_state_from_increment(
                state, dx, problem, "pressure", self.dpMaxRel, self.dpMaxAbs)
            state = self.cap_property(state, "pressure", self.minimumPressure,
                                      self.maximumPressure)
            rest_vars, _ = self.strip_vars(rest_vars, "pressure")
            for name in rest_vars:
                state, _, _ = self.update_state_from_increment(state, dx,
                                                               problem, name)
        return state, None

    def update_after_convergence(self, state0: State, state: State, dt: float,
                                 driving_forces: Any):
        if self.FacilityModel is not None:
            state, f_report = self.FacilityModel.update_after_convergence(
                state0, state, dt, driving_forces)
        else:
            f_report = None
        if self.AquiferModel is not None:
            state, _ = self.AquiferModel.update_after_convergence(
                state0, state, dt, driving_forces)
        state, report = super().update_after_convergence(state0, state, dt,
                                                         driving_forces)
        report["FacilityReport"] = f_report
        if "sMax" in state:
            s = _saturation_matrix(state["s"])
            if np.shape(state["sMax"]) != s.shape:
                state["sMax"] = s.copy()
            else:
                state["sMax"] = np.maximum(state["sMax"], s)
        state.pop("FacilityState", None)
        return state, report

    def setup_operators(self, G: Any = None, rock: Any = None, **options: Any):
        from pymrst.modules.ad_core.utils.operators import setup_operators_tpfa
        G = self.G if G is None else G
        rock = self.rock if rock is None else rock
        self.operators = setup_operators_tpfa(G, rock, **options)
        return self

    def get_convergence_values(self, problem: Any, *args: Any):
        if self.useCNVConvergence:
            from pymrst.modules.ad_core.ad_oo.get_convergence_values_cnv import (
                get_convergence_values_cnv,
            )
            v_cnv, names_cnv, tol_cnv, is_cnv = get_convergence_values_cnv(
                self, problem)
            v_wells, tol_wells, names_wells, is_well = \
                self.FacilityModel.get_facility_convergence_values(problem)
            rest = ~(np.asarray(is_cnv) | np.asarray(is_well))
            values_all = problem.norm(np.inf)
            names_rest = [n for n, r in zip(problem.equation_names, rest) if r]
            values_rest = values_all[rest]
            tol_rest = np.full(values_rest.shape, self.nonlinearTolerance)
            values = np.concatenate([v_cnv, v_wells, values_rest])
            tolerances = np.concatenate([tol_cnv, tol_wells, tol_rest])
            names = list(names_cnv) + list(names_wells) + names_rest
            return values, tolerances, names
        values, tolerances, names = super().get_convergence_values(problem, *args)
        values = np.array(values, dtype=float)
        tolerances = np.array(tolerances, dtype=float)
        if self.FacilityModel is not None:
            v_wells, tol_wells, names_wells, is_well = \
                self.FacilityModel.get_facility_convergence_values(problem)
            is_well = np.asarray(is_well, dtype=bool)
            if np.any(is_well):
                tolerances[is_well] = tol_wells
                values[is_well] = v_wells
                for slot, name in zip(np.flatnonzero(is_well), names_wells):
                    names[slot] = name
        return values, tolerances, names

    # -- variable fields ----------------------------------------------------

    def get_variable_field(self, name: str, throw_error: bool = True):
        key = name.lower()
        if key in ("t", "temperature"):
            return "T", WHOLE
        if key in ("swmax", "somax", "sgmax"):
            return "sMax", self.sat_var_index(key[:2])
        if key == "smax":
            return "sMax", WHOLE
        if key in ("sw", "water"):
            return "s", self.sat_var_index("sw")
        if key in ("so", "oil"):
            return "s", self.sat_var_index("so")
        if key in ("sg", "gas"):
            return "s", self.sat_var_index("sg")
        if key in ("s", "sat", "saturation"):
            return "s", WHOLE
        if key in ("pressure", "p"):
            return "pressure", WHOLE
        if key == "flux":
            return "flux", WHOLE
        if key == "aquifersol":
            return "aquiferSol", WHOLE
        if key == "wellsol":
            return "wellSol", WHOLE
        return super().get_variable_field(name, throw_error)

    def get_component_names(self) -> List[str]:
        return []

    def get_valid_driving_forces(self) -> Dict[str, Any]:
        forces = super().get_valid_driving_forces()
        forces.update(W=None, bc=None, src=None)
        return forces

    def split_primary_variables(self, variables: Sequence[str]):
        well_names = ([] if self.FacilityModel is None
                      else self.FacilityModel.get_primary_variable_names())
        sat_names = [n.lower() for n in self.get_saturation_var_names()]
        well_lower = [n.lower() for n in well_names]
        is_sat = [v.lower() in sat_names for v in variables]
        is_well = [v.lower() in well_lower for v in variables]
        well_vars = [v for v, w in zip(variables, is_well) if w]
        sat_vars = [v for v, s in zip(variables, is_sat) if s]
        rest_vars = [v for v, s, w in zip(variables, is_sat, is_well)
                     if not (s or w)]
        return rest_vars, sat_vars, well_vars

    # -- phases -------------------------------------------------------------

    def get_active_phases(self) -> np.ndarray:
        return np.array([self.water, self.oil, self.gas], dtype=bool)

    def get_phase_names(self) -> str:
        """Active phase letters, e.g. `'WOG'` (MRST returns a char array)."""
        return "".join(l for l, on in zip(PHASE_LETTERS, self.get_active_phases())
                       if on)

    def get_long_phase_names(self) -> List[str]:
        """MRST's second `getPhaseNames` output: `{'water', 'oil', ...}`."""
        return [l for l, on in zip(LONG_PHASE_NAMES, self.get_active_phases())
                if on]

    def get_phase_indices(self) -> np.ndarray:
        w, o, g = (int(x) for x in (self.water, self.oil, self.gas))
        indices = np.array([w, w + o, w + o + g]) - 1
        indices[~self.get_active_phases()] = -1
        return indices

    def get_phase_index(self, *names: str):
        active = self.get_phase_names()
        out = [active.find(n) if n in active else None for n in names]
        return out[0] if len(out) == 1 else out

    def get_number_of_components(self) -> int:
        return len(self.Components)

    def get_number_of_phases(self) -> int:
        return int(np.count_nonzero(self.get_active_phases()))

    def update_saturations(self, state: State, dx: Sequence[Any], problem: Any,
                           sat_vars: Optional[Sequence[str]] = None) -> State:
        """Apply saturation increments, then keep every row physical.

        One saturation is usually not a primary variable: it fills the rest
        so the row sums to one. A row whose update leaves [0, 1] - or that the
        relative permeability's immobile chop touched - is clipped and
        renormalised.
        """
        if sat_vars is None:
            _, sat_vars, _ = self.split_primary_variables(problem.primary_variables)
        if not sat_vars:
            return state
        state_init = State(dict(state))
        saturations = [n.lower() for n in self.get_saturation_var_names()]
        fill = [n for n in saturations if n not in [v.lower() for v in sat_vars]]
        assert len(fill) in (0, 1)
        solved_for = [n not in fill for n in saturations]
        ds = np.zeros((self.G.cells.num, len(saturations)))
        remainder = 0.0
        for i, name in enumerate(saturations):
            if solved_for[i]:
                v = np.asarray(value(self.get_increment(dx, problem, name)),
                               dtype=float)
                ds[:, i] = v
                if fill:
                    remainder = remainder - v
        ds[:, [not f for f in solved_for]] = np.reshape(remainder, (-1, 1)) \
            if fill else ds[:, [not f for f in solved_for]]
        state, _, _ = self.update_state_from_increment(state, ds, problem, "s",
                                                       np.inf, self.dsMaxAbs)
        if self.FlowPropertyFunctions is None:
            chopped = np.zeros(self.G.cells.num, dtype=bool)
        else:
            kr = self.FlowPropertyFunctions.get_state_function("RelativePermeability")
            state, chopped = kr.apply_immobile_chop(self, state, state_init)
        s = np.array(_saturation_matrix(state["s"]), dtype=float, copy=True)
        if fill:
            bad = np.any((s > 1) | (s < 0), axis=1) | np.asarray(chopped)
            if np.any(bad):
                rows = np.clip(s[bad], 0.0, 1.0)
                s[bad] = rows/rows.sum(axis=1, keepdims=True)
        else:
            s = np.maximum(s, 0.0)
        state["s"] = s
        return state

    # -- storing diagnostics on the state -----------------------------------

    def set_phase_data(self, state: State, data: Sequence[Any], fld: str,
                       subs: Any = WHOLE) -> State:
        active = self.get_active_phases()
        state[fld] = _grow_columns(state[fld], int(np.count_nonzero(active)))
        column = 0
        for i, item in enumerate(data):
            if active[i]:
                if isinstance(subs, str) and subs == WHOLE:
                    state[fld][:, column] = item
                else:
                    state[fld][subs, column] = item
                column += 1
        return state

    def store_fluxes(self, state: State, v_w: Any, v_o: Any, v_g: Any) -> State:
        internal = np.asarray(self.operators.internalConn, dtype=bool)
        state["flux"] = np.zeros((internal.size,
                                  int(self.get_active_phases().sum())))
        fluxes = [_value_or_empty(v) for v in (v_w, v_o, v_g)]
        return self.set_phase_data(state, fluxes, "flux", internal)

    def store_boundary_fluxes(self, state: State, q_w: Any, q_o: Any, q_g: Any,
                              forces: Any) -> State:
        bc = _field(forces, "bc")
        if bc is None or np.size(_field(bc, "face", [])) == 0:
            return state
        faces = np.asarray(bc.face, dtype=np.int64).ravel()
        sign = 1 - 2*(np.asarray(self.G.faces.neighbors)[faces, 1] < 0)
        fluxes = [None if item is None or np.size(value(item)) == 0
                  else np.asarray(value(item), dtype=float)*sign
                  for item in (q_w, q_o, q_g)]
        if "flux" not in state:
            state["flux"] = np.zeros((self.G.faces.num,
                                      int(self.get_active_phases().sum())))
        return self.set_phase_data(state, fluxes, "flux", faces)

    def store_mobilities(self, state: State, mob_w: Any, mob_o: Any,
                         mob_g: Any) -> State:
        state["mob"] = np.zeros((self.G.cells.num,
                                 int(self.get_active_phases().sum())))
        return self.set_phase_data(state, [_value_or_empty(m) for m in
                                           (mob_w, mob_o, mob_g)], "mob")

    def store_upstream_indices(self, state: State, upcw: Any, upco: Any,
                               upcg: Any) -> State:
        n_interfaces = np.shape(self.operators.N)[0]
        state["upstreamFlag"] = np.zeros(
            (n_interfaces, int(self.get_active_phases().sum())), dtype=bool)
        return self.set_phase_data(state, [upcw, upco, upcg], "upstreamFlag")

    def store_density(self, state: State, rho_w: Any, rho_o: Any,
                      rho_g: Any) -> State:
        state["rho"] = np.zeros((self.G.cells.num,
                                 int(self.get_active_phases().sum())))
        return self.set_phase_data(state, [_value_or_empty(r) for r in
                                           (rho_w, rho_o, rho_g)], "rho")

    def storebfactors(self, state: State, b_w: Any, b_o: Any, b_g: Any) -> State:
        state["bfactor"] = np.zeros((self.G.cells.num,
                                     int(self.get_active_phases().sum())))
        return self.set_phase_data(state, [_value_or_empty(b) for b in
                                           (b_w, b_o, b_g)], "bfactor")

    def sat_var_index(self, name: str) -> Optional[int]:
        names = [n.lower() for n in self.get_saturation_var_names()]
        return names.index(name.lower()) if name.lower() in names else None

    def comp_var_index(self, name: str) -> Optional[int]:
        names = [n.lower() for n in getattr(self, "componentVarNames", [])]
        return names.index(name.lower()) if name.lower() in names else None

    def evaluate_rel_perm(self, sat: Sequence[Any], *args: Any):
        nph = self.get_number_of_phases()
        assert nph == len(sat), (
            "The number of saturations must equal the number of active phases.")
        names = self.get_phase_names()
        if nph > 1:
            return getattr(self, f"rel_perm_{names.lower()}")(*sat, self.fluid,
                                                              *args)
        return [getattr(self.fluid, f"kr{names}")(*sat, *args)]

    def get_gravity_vector(self) -> np.ndarray:
        dims = getattr(self.G, "griddim", np.size(self.gravity))
        return np.asarray(self.gravity, dtype=float)[:dims]

    def get_gravity_gradient(self) -> Any:
        gdz = getattr(self.operators, "gdz", None)
        if gdz is not None:
            return gdz
        g = self.get_gravity_vector()
        centroids = np.asarray(self.G.cells.centroids, dtype=float)
        # MRST: `Grad(centroids) * g'` - difference first, then the dot
        # product; scaling the depths first rounds differently.
        grad_c = np.column_stack([np.asarray(self.operators.grad(centroids[:, k]), dtype=float)
                                  for k in range(centroids.shape[1])])
        return grad_c @ np.asarray(g, dtype=float).ravel()[:centroids.shape[1]]

    def get_scaling_factors_cpr(self, problem: Any, names: Sequence[str],
                                solver: Any = None) -> List[Any]:
        return [1.0 for _ in names]

    # -- sources, boundary conditions and wells -----------------------------

    def insert_well_equations(self, eqs, names, types, well_sol0, well_sol,
                              well_vars, well_map, p, mob, rho, dissolved,
                              components, dt, opt):
        """Subtract well sources from the cell equations; append the wells'."""
        fm = self.FacilityModel
        if fm.get_number_of_active_wells(well_sol) == 0:
            return list(eqs), list(names), list(types), well_sol, None
        n_ph = self.get_number_of_phases()
        src, wellsys, well_sol = fm.get_well_contributions(
            well_sol0, well_sol, well_vars, well_map, p, mob, rho, dissolved,
            components, dt, _option(opt, "iteration", -1))
        rho_s = self.get_surface_densities()
        wc = np.asarray(src["sourceCells"], dtype=np.int64)
        long_names = self.get_long_phase_names()
        eqs, names, types = list(eqs), list(names), list(types)
        lowered = [n.lower() for n in names]
        for i in range(n_ph):
            if long_names[i] in lowered:
                sub = lowered.index(long_names[i])
                assert types[sub].lower() == "cell", (
                    "Unable to add source terms to equation that is not per cell.")
                eqs[sub] = _subset_minus(eqs[sub],
                                         src["phaseMass"][i]/rho_s[0, i], wc)
        for i, cname in enumerate(self.get_component_names()):
            if cname.lower() in lowered:
                sub = lowered.index(cname.lower())
                assert types[sub].lower() == "cell", (
                    "Unable to add source terms to equation that is not per cell.")
                eqs[sub] = _subset_minus(eqs[sub], src["components"][i], wc)
        eqs = eqs + list(wellsys["wellEquations"]) + [wellsys["controlEquation"]]
        names = names + list(wellsys["names"]) + ["closureWells"]
        types = types + list(wellsys["types"]) + ["well"]
        return eqs, names, types, well_sol, src

    def add_boundary_conditions_and_sources(self, eqs, names, types, state, p,
                                            s, mob, rho, dissolved, components,
                                            forces):
        """Add `src` and `bc` contributions to the per-phase/component rows.

        Returns `(eqs, state, src)`; `src` holds the source and boundary
        contribution structs, as MRST's third output does.
        """
        from pymrst.modules.ad_core.utils.compute_sources_and_boundary_conditions_ad import (
            compute_sources_and_boundary_conditions_ad,
        )
        src_terms, bnd_cond = compute_sources_and_boundary_conditions_ad(
            self, p, s, mob, rho, dissolved, forces)
        long_names = self.get_long_phase_names()
        rho_s = self.get_surface_densities()
        is_mass = self.isGenericReservoirModel
        eqs = list(eqs)
        lowered = [n.lower() for n in names]
        bc_force = _field(forces, "bc")
        for i in range(len(s)):
            if long_names[i] not in lowered:
                continue
            sub = lowered.index(long_names[i])
            assert types[sub].lower() == "cell", (
                "Unable to add source terms to equation that is not per cell.")
            scale = 1.0 if is_mass else rho_s[0, i]
            sc = np.asarray(src_terms.sourceCells, dtype=np.int64)
            if sc.size:
                eqs[sub] = _subset_minus(eqs[sub], src_terms.phaseMass[i]/scale, sc)
            if bc_force is not None and getattr(bc_force, "phaseMass", None) is not None:
                bnd_cond.phaseMass[i] = np.asarray(bc_force.phaseMass)[:, i]
            bc_cells = np.asarray(bnd_cond.sourceCells, dtype=np.int64)
            if bc_cells.size:
                if bnd_cond.mapping is None:
                    q = bnd_cond.phaseMass[i]/scale
                else:
                    q = ad_matmul(bnd_cond.mapping, bnd_cond.phaseMass[i])/scale
                eqs[sub] = _subset_minus(eqs[sub], q, bc_cells)
        if self.outputFluxes:
            volumes: List[Any] = [None, None, None]
            for slot, item in zip(np.flatnonzero(self.get_active_phases()),
                                  bnd_cond.phaseVolume):
                volumes[slot] = item
            state = self.store_boundary_fluxes(state, *volumes, forces)
        for i, name in enumerate(self.get_component_names()):
            if name.lower() in long_names:
                continue
            if name.lower() in lowered:
                sub = lowered.index(name.lower())
                eq = eqs[sub]
                assert types[sub].lower() == "cell", (
                    "Unable to add source terms to equation that is not per cell.")
            else:
                sub, eq = None, np.zeros(self.G.cells.num)
            component = components[i]
            eq, bnd_cond = self.add_component_contributions(
                name, eq, component, bnd_cond, bc_force)
            eq, src_terms = self.add_component_contributions(
                name, eq, component, src_terms, _field(forces, "src"))
            if sub is not None:
                eqs[sub] = eq
        return eqs, state, {"src": src_terms, "bc": bnd_cond}

    def add_component_contributions(self, cname, eq, component, src, force):
        raise NotImplementedError(
            "This function is not valid for the base class. It should be "
            "implemented for the derived classes that contain components")

    def get_surface_densities(self, regions: Any = None,
                              phases: Any = None) -> np.ndarray:
        """`nreg x nph` surface densities, one row per PVT region.

        `regions` is a region map as stored on the state functions - MRST's
        one-based PVTNUM - so it selects rows `regions - 1`.
        """
        names = self.get_phase_names()
        if phases is not None:
            names = "".join(np.array(list(names))[phases])
        columns = [np.atleast_1d(np.asarray(value(getattr(self.fluid,
                                                          f"rho{n}S")),
                                            dtype=float)).ravel()
                   for n in names]
        rho_s = np.column_stack(columns)
        if regions is not None and np.size(regions):
            rho_s = rho_s[np.asarray(regions, dtype=np.int64).ravel() - 1]
        return rho_s

    def get_extra_well_contributions(self, well, well_sol0, well_sol, q_s, bh,
                                     packed, q_mass, q_vol, dt, iteration):
        return [], [], [], well_sol

    def get_extra_well_equation_names(self):
        return [], []

    def get_extra_well_primary_variable_names(self) -> List[str]:
        return []

    def insert_sources(self, eqs: List[Any], src: Any) -> List[Any]:
        cells = np.asarray(_field(src, "cells", []), dtype=np.int64)
        if cells.size == 0:
            return eqs
        return [_subset_minus(eq, src.value[i], cells) for i, eq in enumerate(eqs)]

    # -- relative permeability combinations (MRST statics) ------------------

    @staticmethod
    def rel_perm_wog(sw, so, sg, f, *args):
        """Stone-I-style mix of oil-water and oil-gas oil curves."""
        swcon = 0.0
        if getattr(f, "sWcon", None) is not None:
            if not args or np.size(f.sWcon) == 1:
                swcon = f.sWcon
            else:
                assert args[0] == "cellInx"
                swcon = np.asarray(f.sWcon)[args[1]]
        swcon = np.minimum(swcon, np.asarray(value(sw)) - 1e-5)
        d = sg + sw - swcon
        ww = (sw - swcon)/d
        kr_w = f.krW(sw, *args)
        wg = 1 - ww
        kr_g = f.krG(sg, *args)
        krow = f.krOW(so, *args)
        krog = f.krOG(so, *args)
        kr_o = wg*krog + ww*krow
        return kr_w, kr_o, kr_g

    @staticmethod
    def rel_perm_wo(sw, so, f, *args):
        kr_w = f.krW(sw, *args)
        kr_o = f.krO(so, *args) if getattr(f, "krO", None) is not None \
            else f.krOW(so, *args)
        return kr_w, kr_o

    @staticmethod
    def rel_perm_og(so, sg, f, *args):
        kr_g = f.krG(sg, *args)
        kr_o = f.krO(so, *args) if getattr(f, "krO", None) is not None \
            else f.krOG(so, *args)
        return kr_o, kr_g

    @staticmethod
    def rel_perm_wg(sw, sg, f, *args):
        return f.krW(sw, *args), f.krG(sg, *args)

    @staticmethod
    def adjust_step_from_sat_bounds(s: np.ndarray, ds: np.ndarray) -> np.ndarray:
        """Shorten each row's step so its worst entry lands on 0 or 1."""
        tmp = s + ds
        violate = np.maximum(np.maximum(tmp - 1, 0), np.abs(np.minimum(tmp, 0)))
        worst = violate.max(axis=1)
        jj = violate.argmax(axis=1)
        bad = worst > 0
        if np.any(bad):
            w = np.ones(s.shape[0])
            for i in range(s.shape[1]):
                ind = bad & (jj == i)
                dworst = np.abs(ds[ind, i])
                w[ind] = (dworst - worst[ind])/dworst
            ds = ds.copy()
            ds[bad, :] = ds[bad, :]*w[bad, None]
        return ds

    # MRST spellings
    getSaturationVarNames = get_saturation_var_names
    setupOperators = setup_operators
    splitPrimaryVariables = split_primary_variables
    getActivePhases = get_active_phases
    getPhaseNames = get_phase_names
    getPhaseIndices = get_phase_indices
    getPhaseIndex = get_phase_index
    getNumberOfComponents = get_number_of_components
    getNumberOfPhases = get_number_of_phases
    getComponentNames = get_component_names
    updateSaturations = update_saturations
    setPhaseData = set_phase_data
    storeFluxes = store_fluxes
    storeBoundaryFluxes = store_boundary_fluxes
    storeMobilities = store_mobilities
    storeUpstreamIndices = store_upstream_indices
    storeDensity = store_density
    satVarIndex = sat_var_index
    compVarIndex = comp_var_index
    evaluateRelPerm = evaluate_rel_perm
    getGravityVector = get_gravity_vector
    getGravityGradient = get_gravity_gradient
    getScalingFactorsCPR = get_scaling_factors_cpr
    insertWellEquations = insert_well_equations
    addBoundaryConditionsAndSources = add_boundary_conditions_and_sources
    addComponentContributions = add_component_contributions
    getSurfaceDensities = get_surface_densities
    getExtraWellContributions = get_extra_well_contributions
    getExtraWellEquationNames = get_extra_well_equation_names
    getExtraWellPrimaryVariableNames = get_extra_well_primary_variable_names
    insertSources = insert_sources
    relPermWOG = rel_perm_wog
    relPermWO = rel_perm_wo
    relPermOG = rel_perm_og
    relPermWG = rel_perm_wg
    adjustStepFromSatBounds = adjust_step_from_sat_bounds
    validateState = validate_state
    validateModel = validate_model
    validateDrivingForces = validate_driving_forces
    getMaximumTimestep = get_maximum_timestep
    prepareReportstep = prepare_report_step
    prepareTimestep = prepare_timestep
    setupStateFunctionGroupings = setup_state_function_groupings
    setStateFunctionEvaluationMode = set_state_function_evaluation_mode
    removeStateFunctionGroupings = remove_state_function_groupings
    getStateFunctionGroupings = get_state_function_groupings
    stepFunction = step_function
    updateForChangedControls = update_for_changed_controls
    updateState = update_state
    updateAfterConvergence = update_after_convergence
    getConvergenceValues = get_convergence_values
    getVariableField = get_variable_field
    getValidDrivingForces = get_valid_driving_forces


def _grow_columns(array: Any, ncols: int) -> np.ndarray:
    """MATLAB grows a matrix when a column past its end is assigned:
    `initResSol`'s one-column `flux` becomes one column per phase."""
    array = np.asarray(array)
    if array.ndim == 1:
        array = array[:, None]
    if array.shape[1] < ncols:
        pad = np.zeros((array.shape[0], ncols - array.shape[1]), dtype=array.dtype)
        array = np.hstack([array, pad])
    return array


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _wells_of(forces: Any) -> List[Any]:
    wells = _field(forces, "W")
    if wells is None:
        return []
    return list(wells) if isinstance(wells, (list, tuple)) else [wells]


def _option(opt: Any, name: str, default: Any) -> Any:
    found = _field(opt, name, None)
    return default if found is None else found


def _saturation_matrix(s: Any) -> np.ndarray:
    """`s` as a numeric matrix, whether stored as a matrix or AD columns."""
    if isinstance(s, list):
        return np.column_stack([np.asarray(value(c), dtype=float).ravel()
                                for c in s])
    array = np.asarray(value(s), dtype=float)
    return array.reshape(-1, 1) if array.ndim == 1 else array


def _value_or_empty(item: Any) -> Any:
    if item is None:
        return None
    return np.asarray(value(item), dtype=float)


def _subset_minus(eq: Any, contribution: Any, cells: np.ndarray) -> Any:
    """MRST `eq(cells) = eq(cells) - v`, keeping derivatives on either side.

    An assignment, as in MATLAB: a repeated cell keeps its last write rather
    than summing. MRST never relies on repeats here - wells merge repeated
    perforations and boundary conditions go through `mapping` - so neither
    does this.
    """
    from pymrst.modules.ad_core.utils.double_to_adi import double_to_adi
    cells = np.asarray(cells, dtype=np.int64)
    if not hasattr(eq, "jac") and hasattr(contribution, "jac"):
        eq = double_to_adi(np.asarray(eq, dtype=float), contribution)
    # A repeated cell (a corner cell with two boundary faces) is written once
    # per occurrence in MATLAB, each write carrying the same mapped total, so
    # the last write leaves value and derivative alike. Assigning AD rows
    # with a repeated index *sums* their Jacobian rows instead, doubling the
    # corner derivatives; writing each distinct cell once is MATLAB's result.
    # The last occurrence is used, as MATLAB's last write is the one kept.
    reversed_first = np.unique(cells[::-1], return_index=True)[1]
    keep = cells.size - 1 - reversed_first
    cells_u = cells[keep]
    if np.size(value(contribution)) == cells.size:
        contribution = contribution[keep]
    if hasattr(eq, "jac"):
        out = eq + 0.0
        out[cells_u] = eq[cells_u] - contribution
        return out
    out = np.array(eq, dtype=float, copy=True)
    out[cells_u] = out[cells_u] - np.asarray(contribution, dtype=float).ravel()
    return out


__all__ = ["ReservoirModel", "LONG_PHASE_NAMES", "PHASE_LETTERS"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_phase_bookkeeping_follows_the_active_flags():
        model = toy_model("WG", nx=2)
        assert model.get_phase_names() == "WG"
        assert model.get_long_phase_names() == ["water", "gas"]
        np.testing.assert_array_equal(model.get_phase_indices(), [0, -1, 1])
        assert model.get_phase_index("G") == 1
        print("ok test_phase_bookkeeping_follows_the_active_flags")

    def test_saturation_names_read_columns_of_s():
        model = toy_model("WOG", nx=2)
        state = toy_state(model, s=np.array([[0.1, 0.6, 0.3], [0.2, 0.5, 0.3]]))
        np.testing.assert_allclose(model.get_prop(state, "oil"), [0.6, 0.5])
        np.testing.assert_allclose(model.get_prop(state, "sg"), [0.3, 0.3])
        print("ok test_saturation_names_read_columns_of_s")

    def test_surface_densities_are_one_row_per_region():
        model = toy_model("WO", nx=2)
        np.testing.assert_allclose(model.get_surface_densities(), [[1000.0, 700.0]])
        rho = model.get_surface_densities(np.array([1, 1, 1]))
        assert rho.shape == (3, 2)
        print("ok test_surface_densities_are_one_row_per_region")

    def test_saturation_update_fills_the_unsolved_phase_and_limits_the_step():
        model = toy_model("WO", nx=2)
        state = toy_state(model, s=np.array([[0.2, 0.8], [0.5, 0.5]]))
        problem = SimpleNamespace(primary_variables=["sW"],
                                  index_of_primary_variable=lambda n: 0 if n.lower() == "sw" else None)
        state = model.update_saturations(state, [np.array([0.5, 0.1])], problem)
        # Row one asked for +0.5; dsMaxAbs = 0.2 caps it.
        np.testing.assert_allclose(state["s"], [[0.4, 0.6], [0.6, 0.4]])
        print("ok test_saturation_update_fills_the_unsolved_phase_and_limits_the_step")

    def test_step_from_sat_bounds_lands_the_worst_entry_on_the_bound():
        ds = ReservoirModel.adjust_step_from_sat_bounds(np.array([[0.9, 0.1]]),
                                                        np.array([[0.2, -0.2]]))
        np.testing.assert_allclose(ds, [[0.1, -0.1]])
        print("ok test_step_from_sat_bounds_lands_the_worst_entry_on_the_bound")

    test_phase_bookkeeping_follows_the_active_flags()
    test_saturation_names_read_columns_of_s()
    test_surface_densities_are_one_row_per_region()
    test_saturation_update_fills_the_unsolved_phase_and_limits_the_step()
    test_step_from_sat_bounds_lands_the_worst_entry_on_the_bound()
    print("\nAll tests passed!")
