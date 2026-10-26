"""MRST `ad-core/models/PhysicalModel.m` base model protocol.

MRST's `PhysicalModel` carries 49 methods; this file carried 4, and the parity
ledger called it `ported` because a file of the right name existed. The AD-OO
machinery MRST centralises here - property access by name, update limiting,
state-function container lifecycle, schedule validation - was not ported, so
every concrete model that needed a piece of it grew its own, and the only place
update limiting existed at all was `dual_porosity/dp_model.py`.

**This module is additive.** Every method below is new; nothing in PyMRST
called any of them before, so no existing result can move. The four that were
already here - `get_props`, `validate_model`, and the three abstract hooks -
keep their behaviour exactly, including `update_state` staying abstract rather
than adopting MRST's default loop. Adopting that default would change what a
subclass without an override does, which belongs to the migration phase.

Two methods are structurally complete but depend on machinery that is itself
still shallow, and say so where they are defined: `get_state_ad` seeds AD
through `pymrst.utils.adi` because `AutoDiffBackend` does not dispatch, and
`solve_adjoint` needs an adjoint-equation protocol no model implements yet.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.utils.adi import value


class PhysicalModel:
    """The AD-OO protocol: properties, equations, updates, convergence."""

    def __init__(self):
        self.groupings: List[str] = []
        # MRST: 'lazy' evaluates a state function when it is asked for; 'full'
        # pre-evaluates the whole graph once per state.
        self.state_function_evaluation_mode: str = "lazy"
        self.state_function_graph: Optional[List[Tuple[str, str]]] = None
        self.nonlinear_tolerance: float = 1e-6
        self.verbose: bool = False

    # -- property access ----------------------------------------------------

    def get_variable_field(self, name: str, strict: bool = True):
        """`(field, index)` for a named variable, or `(None, None)`.

        MRST's base knows nothing and raises when asked strictly; a model that
        owns a variable overrides this. `get_prop` falls back to the state
        function groupings when it returns nothing, which is how a property
        that is *computed* rather than stored gets reached.
        """
        if strict:
            raise KeyError(
                f"state variable {name!r} is not known to "
                f"{type(self).__name__}")
        return None, None

    def get_prop(self, state: State, name: str) -> Any:
        """One property, from the state's fields or from a grouping.

        Order matters and is MRST's: the variable field is tried first, so a
        stored primary variable is never shadowed by a state function that
        happens to share its name.
        """
        field, index = self.get_variable_field(name, strict=False)
        if field is None:
            return self._get_one(state, name)
        stored = state[field]
        if index is None or isinstance(index, slice):
            return stored
        if isinstance(stored, (list, tuple)):
            return stored[index]
        array = np.asarray(value(stored))
        return stored if array.ndim == 1 else array[:, index]

    def get_props(self, state: State, *names: str) -> Any:
        """One or several properties, from whichever grouping declares them."""
        out = [self._get_one(state, n) for n in names]
        return out[0] if len(out) == 1 else out

    def _get_one(self, state: State, name: str) -> Any:
        for gname in self.groupings:
            group = getattr(self, gname)
            if group.has_state_function(name):
                return group.get(self, state, name)
        if name in state:
            return state[name]
        raise KeyError(
            f"'{name}' is neither a state variable nor a state function of "
            f"{type(self).__name__}"
        )

    def set_prop(self, state: State, name: str, new_value: Any) -> State:
        """Write a named property back into the state.

        Mirrors MRST's column handling: a field stored as one column is
        replaced wholesale, a multi-column field has a single column written.
        """
        field, index = self.get_variable_field(name, strict=False)
        if field is None:
            field, index = name, slice(None)
        if field not in state or index is None or isinstance(index, slice):
            state[field] = new_value
            return state
        stored = state[field]
        if isinstance(stored, list):
            stored[index] = new_value
            return state
        array = np.asarray(value(stored))
        if array.ndim == 1:
            state[field] = new_value
        else:
            array = array.copy()
            array[:, index] = value(new_value)
            state[field] = array
        return state

    def set_props(self, state: State, names: Sequence[str],
                  values: Sequence[Any]) -> State:
        if len(names) != len(values):
            raise ValueError(f"{len(names)} names but {len(values)} values")
        for name, item in zip(names, values):
            state = self.set_prop(state, name, item)
        return state

    def increment_prop(self, state: State, name: str, increment: Any) -> State:
        return self.set_prop(state, name,
                             self.get_prop(state, name) + increment)

    def cap_property(self, state: State, name: str, minvalue: float,
                     maxvalue: Optional[float] = None) -> State:
        """Clamp a property into a range.

        The blunt instrument behind saturations staying physical. `minvalue`
        may be `-inf`, and `maxvalue` omitted for a one-sided bound.
        """
        capped = np.maximum(minvalue, value(self.get_prop(state, name)))
        if maxvalue is not None:
            capped = np.minimum(capped, maxvalue)
        return self.set_prop(state, name, capped)

    def check_property(self, state: State, prop: str, n_el, dim) -> None:
        """Assert a property's shape, naming what was expected and what is there.

        MRST raises here rather than letting a wrong-sized field reach an
        assembly, where it surfaces as a shape error in unrelated code.
        """
        if isinstance(dim, (list, tuple, np.ndarray)):
            if len(n_el) != len(dim):
                raise ValueError("n_el and dim must have the same length")
            for count, axis in zip(n_el, dim):
                self.check_property(state, prop, count, axis)
            return

        field, _ = self.get_variable_field(prop, strict=False)
        field = field or prop
        if field not in state:
            raise KeyError(
                f'field "{field}" missing: {prop} must be supplied for model '
                f'"{type(self).__name__}"')
        # MRST stores a scalar field as an n-by-1 column, so `size(f, 1)` is
        # the cell count. `np.atleast_2d` would make a 1-D array 1-by-n and
        # report one row for every model - the check would pass for any cell
        # count and fail for the true one.
        array = np.asarray(value(state[field]))
        if array.ndim == 1:
            array = array[:, None]
        axis = 0 if dim == 1 else 1
        actual = array.shape[axis] if axis < array.ndim else 1
        if actual != n_el:
            which = ("rows" if dim == 1 else
                     "columns" if dim == 2 else f"dimension {dim}")
            raise ValueError(
                f'dimension mismatch for {which} of property "{prop}" '
                f'(state.{field}): expected {n_el}, state had {actual}')

    # -- Newton updates -----------------------------------------------------
    #
    # The group that existed nowhere in PyMRST. Without it nothing stops a
    # Newton increment from driving a saturation outside [0, 1]; with tabulated
    # properties the next evaluation then extrapolates off the end of a table,
    # and the failure appears several iterations later as non-convergence
    # rather than as the bad update that caused it.

    @staticmethod
    def limit_update_relative(dv, val, max_rel_change):
        """Scale an increment so no entry moves more than a fraction of itself.

        The scaling is *per row and uniform across columns* - one factor for a
        whole row - so a multi-component update keeps its direction and only
        changes length. Scaling each column on its own would rotate the update,
        which is a different step, not a shorter one.
        """
        dv = np.atleast_2d(np.asarray(value(dv), dtype=float))
        val = np.atleast_2d(np.asarray(value(val), dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            biggest = np.max(np.abs(dv/val), axis=1)
            change = np.minimum(max_rel_change/biggest, 1.0)
        change = np.where(np.isfinite(change), change, 1.0)
        return dv*change[:, None], change

    @staticmethod
    def limit_update_absolute(dv, max_abs_change):
        """Scale an increment so no entry moves more than a fixed amount."""
        dv = np.atleast_2d(np.asarray(value(dv), dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            biggest = np.max(np.abs(dv), axis=1)
            change = np.minimum(max_abs_change/biggest, 1.0)
        change = np.where(np.isfinite(change), change, 1.0)
        return dv*change[:, None], change

    @staticmethod
    def index_of_primary_variable(problem, name: str) -> Optional[int]:
        names = (getattr(problem, "primary_variables", None)
                 or getattr(problem, "var_names", None) or [])
        lowered = [str(n).lower() for n in names]
        key = str(name).lower()
        return lowered.index(key) if key in lowered else None

    def get_increment(self, dx, problem, name: str):
        """The increment for one named primary variable, or zero.

        Zero rather than an error when the variable is absent: MRST's models
        ask for variables a particular problem may not carry, and a missing
        increment means "this one did not move".
        """
        if isinstance(dx, dict):
            return dx.get(name, 0.0)
        index = self.index_of_primary_variable(problem, name)
        if index is None:
            return 0.0
        return dx[index]

    def update_state_from_increment(self, state: State, dx, problem,
                                    name: str,
                                    rel_change_max: Optional[float] = None,
                                    abs_change_max: Optional[float] = None):
        """Apply one named increment, optionally limited.

        Returns `(state, value, previous)`. When both limits are given the
        *smaller* scaling wins, so the update satisfies both rather than
        whichever was applied last.
        """
        dv = (dx if isinstance(dx, (int, float, np.ndarray))
              else self.get_increment(dx, problem, name))
        previous = self.get_prop(state, name)

        has_rel = (rel_change_max is not None
                   and np.all(np.isfinite(rel_change_max)))
        has_abs = (abs_change_max is not None
                   and np.all(np.isfinite(abs_change_max)))
        if not (has_rel or has_abs):
            updated = previous + dv
        else:
            flat = np.ndim(value(dv)) < 2
            dv2 = np.atleast_2d(np.asarray(value(dv), dtype=float))
            change = np.ones(dv2.shape[0])
            if has_rel:
                _, rel = self.limit_update_relative(dv2, previous, rel_change_max)
                change = np.minimum(change, rel)
            if has_abs:
                _, absolute = self.limit_update_absolute(dv2, abs_change_max)
                change = np.minimum(change, absolute)
            scaled = dv2*change[:, None]
            updated = previous + (scaled.ravel() if flat else scaled)
        return self.set_prop(state, name, updated), updated, previous

    @staticmethod
    def strip_vars(variables: Sequence[str], names: Sequence[str]):
        """Drop named variables, case-insensitively. Returns `(kept, removed)`."""
        drop = {str(n).lower() for n in names}
        removed = [str(v).lower() in drop for v in variables]
        kept = [v for v, gone in zip(variables, removed) if not gone]
        return kept, removed

    # -- state-function groupings -------------------------------------------

    def get_state_function_groupings(self) -> List[Any]:
        """Every grouping this model owns, in declaration order."""
        return [getattr(self, name) for name in self.groupings
                if getattr(self, name, None) is not None]

    def init_state_function_containers(self, state: State) -> State:
        for group in self.get_state_function_groupings():
            state.containers.setdefault(group.struct_name, {})
        return state

    def set_state_function_evaluation_mode(self, mode: str) -> "PhysicalModel":
        if str(mode).lower() not in ("lazy", "full"):
            raise ValueError(
                f"unknown evaluation mode {mode!r}; expected 'lazy' (evaluate "
                f"when needed) or 'full' (pre-evaluate everything)")
        self.state_function_evaluation_mode = str(mode).lower()
        return self

    def get_state_function_evaluation_mode(self) -> str:
        return self.state_function_evaluation_mode

    def setup_state_function_graph(self, filter_names: Sequence[str] = ()):
        """Build and cache the evaluation order across every grouping.

        Stored as `(grouping, function)` pairs in dependency order, which is
        what `evaluate_all_state_functions` walks. `filter_names` accepts
        `"Grouping.function"`, `"Grouping.*"` or a bare function name, matching
        MRST's filter spelling.
        """
        order: List[Tuple[str, str]] = []
        for gname in self.groupings:
            group = getattr(self, gname, None)
            if group is None:
                continue
            for fname in group.evaluation_order():
                order.append((gname, fname))

        skip = set()
        for entry in filter_names:
            parts = str(entry).split(".")
            for gname, fname in order:
                if len(parts) == 2:
                    if parts[0].lower() == gname.lower() and (
                            parts[1] == "*"
                            or parts[1].lower() == fname.lower()):
                        skip.add((gname, fname))
                elif parts[0].lower() == fname.lower():
                    skip.add((gname, fname))

        self.state_function_graph = [p for p in order if p not in skip]
        return self, self.state_function_graph

    def evaluate_all_state_functions(self, state: State) -> State:
        """Evaluate the whole graph at once, instead of on demand."""
        if self.state_function_graph is None:
            self.setup_state_function_graph()
        state = self.init_state_function_containers(state)
        for gname, fname in self.state_function_graph:
            getattr(self, gname).get(self, state, fname)
        state["evaluated"] = True
        return state

    def check_state_function_dependencies(self) -> None:
        """Every declared dependency resolves. The same checks as
        `validate_model`, under MRST's name, so a caller written against
        MATLAB finds it."""
        self.validate_model()

    def validate_model(self) -> None:
        """Check every declared dependency can actually be resolved."""
        for gname in self.groupings:
            group = getattr(self, gname)
            group.evaluation_order()
            for fname, deps in group.dependency_graph().items():
                for dep in deps:
                    if dep.grouping == "state":
                        continue
                    owner = (group if dep.grouping == "self"
                             else getattr(self, dep.grouping, None))
                    if owner is None:
                        raise KeyError(
                            f"{type(group).__name__}.{fname} depends on "
                            f"grouping '{dep.grouping}', which the model "
                            f"does not have"
                        )
                    if not owner.has_state_function(dep.name):
                        raise KeyError(
                            f"{type(group).__name__}.{fname} depends on "
                            f"'{dep.name}' in {dep.grouping}, which does not "
                            f"define it"
                        )

    # -- state lifecycle ----------------------------------------------------

    def validate_state(self, state: State) -> State:
        if "time" not in state:
            state["time"] = 0.0
        return state

    def reduce_state(self, state: State, remove_containers: bool = True) -> State:
        """Strip a state to stored values, dropping AD and caches.

        What survives between Newton iterations. Leaving a grouping cache on a
        state whose primary variables have since changed is how a stale
        property reaches the next iteration.
        """
        keep = dict(state.containers)
        reduced = State({k: value(v) for k, v in state.items()
                         if k not in ("primaryVariables", "primary_variables",
                                      "evaluated")})
        reduced.containers = {} if remove_containers else keep
        return reduced

    def get_state_ad(self, state: State, init: bool = True):
        """A state whose primary variables carry derivatives.

        Seeded through `pymrst.utils.adi` rather than
        `model.AutoDiffBackend.initVariablesAD`, because `AutoDiffBackend` is
        still a capability descriptor with no dispatch - it records which
        backend was asked for and supplies none. When the backend gains its
        MRST interface, this is the line that changes.
        """
        from pymrst.utils.adi import init_variables_adi

        variables, names, origin = self.get_primary_variables(state)
        if init:
            seeded = init_variables_adi(
                *[np.asarray(value(v), dtype=float).ravel() for v in variables])
            variables = list(seeded) if isinstance(seeded, (list, tuple)) else [seeded]
            state["primaryVariables"] = list(names)
        state = self.init_state_ad(state, variables, names, origin)
        if self.state_function_evaluation_mode == "full":
            state = self.evaluate_all_state_functions(state)
        return state, names, origin

    def get_reverse_state_ad(self, state: State, init: bool = True):
        """The adjoint's state. Identical to `get_state_ad` in MRST, which
        keeps the separate name so a reverse-mode caller reads as reverse
        mode."""
        return self.get_state_ad(state, init)

    def init_state_ad(self, state: State, variables, names, origin=None) -> State:
        for name, var in zip(names, variables):
            state = self.set_prop(state, name, var)
        return self.init_state_function_containers(state)

    def get_primary_variables(self, state: State):
        """`(values, names, origin)`. A concrete model declares its own."""
        raise NotImplementedError(
            f"{type(self).__name__} does not declare primary variables")

    # -- driving forces and schedule ----------------------------------------

    def get_valid_driving_forces(self) -> Dict[str, Any]:
        """The forces this model accepts. Names carry the defaults."""
        return {"W": None, "bc": None, "src": None}

    def get_driving_forces(self, control: Any) -> Dict[str, Any]:
        """A control step's forces, as a mapping."""
        if control is None:
            return {}
        if isinstance(control, dict):
            return dict(control)
        return {k: v for k, v in vars(control).items() if not k.startswith("_")}

    def validate_driving_forces(self, forces: Any, index: int = 1) -> Any:
        """Refuse a force this model does not accept.

        A misspelled force is otherwise silent: carried along, never read, and
        the run proceeds as though nothing had been asked for.
        """
        valid = set(self.get_valid_driving_forces())
        for name in self.get_driving_forces(forces):
            if name not in valid:
                raise KeyError(
                    f'driving force "{name}" is not accepted by '
                    f"{type(self).__name__}; valid forces are {sorted(valid)}")
        return forces

    def validate_schedule(self, schedule: Any) -> Any:
        """Check a schedule is internally consistent before anything runs.

        Every check fails at setup instead of mid-simulation, which is the
        point: a control index past the end of `control` would otherwise raise
        on whichever step first used it, hours in.
        """
        def field(obj, name):
            if isinstance(obj, dict):
                return obj.get(name)
            return getattr(obj, name, None)

        control = field(schedule, "control")
        step = field(schedule, "step")
        if control is None:
            raise ValueError(
                "schedule is missing .control; cannot define driving forces")
        if step is None:
            raise ValueError("schedule is missing .step; cannot define timesteps")
        val = field(step, "val")
        ctrl = field(step, "control")
        if val is None:
            raise ValueError(
                "schedule is missing .step.val; cannot define timesteps")
        if ctrl is None:
            raise ValueError(
                "schedule is missing .step.control; cannot define controls for "
                "each timestep")

        val = np.asarray(val, dtype=float).ravel()
        ctrl = np.asarray(ctrl, dtype=np.int64).ravel()
        if val.size != ctrl.size:
            raise ValueError(
                f"schedule.step.val has {val.size} entries and "
                f"schedule.step.control has {ctrl.size}; they must match")
        if ctrl.size and ctrl.min() < 1:
            raise ValueError(
                "schedule.step.control must be one-based indices into "
                "schedule.control")
        n_control = len(control)
        if ctrl.size and ctrl.max() > n_control:
            raise ValueError(
                f"schedule.control has length {n_control}, but the largest "
                f"control in schedule.step.control was {int(ctrl.max())}")
        if np.any(val < 0):
            raise ValueError("negative timesteps present in schedule.step.val")

        for index in range(n_control):
            self.validate_driving_forces(control[index], index + 1)
        return schedule

    def prepare_report_step(self, state: State, state0: State, dt: float,
                            forces: Any = None):
        """Hook before a report step. MRST revalidates the model against the
        forces that are about to apply."""
        self.validate_model()
        return self, state

    def prepare_timestep(self, state: State, state0: State, dt: float,
                         forces: Any = None):
        return self, state

    def get_maximum_timestep(self, state: State, state0: State, dt: float,
                             forces: Any = None) -> float:
        return float("inf")

    # -- linearized problem and adjoint -------------------------------------

    def setup_linearized_problem(self, eqs, types, names, primary_vars,
                                 state, dt):
        from pymrst.modules.ad_core.solvers.linear_solver_ad import (
            LinearizedProblem,
        )
        # `from_equations`, not the constructor: MRST's argument order is
        # (equations, types, names, primaryVars, state, dt) while PyMRST's
        # `__init__` is (A, b, ...). Calling the constructor with MRST's order
        # put the equations in `A` and the type strings in `b`, and raised on
        # the first use - which nothing did, so it went unnoticed until the
        # ledger drove a test through this path.
        return LinearizedProblem.from_equations(eqs, types, names,
                                                primary_vars, state, dt)

    def solve_adjoint(self, solver, get_state, get_obj, schedule, lam,
                      step_no: int, **kwargs):
        """One backward step of the adjoint sweep.

        Structurally MRST's, and it needs `get_adjoint_equations`, which no
        PyMRST model defines - `ReservoirModel` is itself shallow. Raising here
        names the missing piece rather than pretending; the working two-phase
        adjoint lives in `modules/optimization/adjoint.py` and does not go
        through this protocol.
        """
        raise NotImplementedError(
            f"the AD-OO adjoint protocol needs get_adjoint_equations, which "
            f"{type(self).__name__} does not implement; see "
            f"pymrst.modules.optimization.adjoint for the two-phase adjoint "
            f"that does not use this path")

    # -- equations, updates, convergence (unchanged, still abstract) --------

    def get_equations(self, state0: State, state: State, dt: float):
        raise NotImplementedError

    def update_state(self, state: State, increment: np.ndarray) -> State:
        raise NotImplementedError

    def get_convergence(self, residuals, dt: float) -> Tuple[bool, Dict[str, float]]:
        raise NotImplementedError


# MRST spellings, so code written against the MATLAB names resolves.
PhysicalModel.getProp = PhysicalModel.get_prop
PhysicalModel.getProps = PhysicalModel.get_props
PhysicalModel.setProp = PhysicalModel.set_prop
PhysicalModel.setProps = PhysicalModel.set_props
PhysicalModel.capProperty = PhysicalModel.cap_property
PhysicalModel.checkProperty = PhysicalModel.check_property
PhysicalModel.incrementProp = PhysicalModel.increment_prop
PhysicalModel.getIncrement = PhysicalModel.get_increment
PhysicalModel.updateStateFromIncrement = PhysicalModel.update_state_from_increment
PhysicalModel.limitUpdateRelative = PhysicalModel.limit_update_relative
PhysicalModel.limitUpdateAbsolute = PhysicalModel.limit_update_absolute
PhysicalModel.stripVars = PhysicalModel.strip_vars
PhysicalModel.getStateFunctionGroupings = PhysicalModel.get_state_function_groupings
PhysicalModel.initStateFunctionContainers = PhysicalModel.init_state_function_containers
PhysicalModel.setStateFunctionEvaluationMode = PhysicalModel.set_state_function_evaluation_mode
PhysicalModel.getStateFunctionEvaluationMode = PhysicalModel.get_state_function_evaluation_mode
PhysicalModel.setupStateFunctionGraph = PhysicalModel.setup_state_function_graph
PhysicalModel.evaluateAllStateFunctions = PhysicalModel.evaluate_all_state_functions
PhysicalModel.checkStateFunctionDependencies = PhysicalModel.check_state_function_dependencies
PhysicalModel.validateModel = PhysicalModel.validate_model
PhysicalModel.validateState = PhysicalModel.validate_state
PhysicalModel.reduceState = PhysicalModel.reduce_state
PhysicalModel.getStateAD = PhysicalModel.get_state_ad
PhysicalModel.getReverseStateAD = PhysicalModel.get_reverse_state_ad
PhysicalModel.initStateAD = PhysicalModel.init_state_ad
PhysicalModel.getPrimaryVariables = PhysicalModel.get_primary_variables
PhysicalModel.getValidDrivingForces = PhysicalModel.get_valid_driving_forces
PhysicalModel.getDrivingForces = PhysicalModel.get_driving_forces
PhysicalModel.validateDrivingForces = PhysicalModel.validate_driving_forces
PhysicalModel.validateSchedule = PhysicalModel.validate_schedule
PhysicalModel.prepareReportstep = PhysicalModel.prepare_report_step
PhysicalModel.prepareTimestep = PhysicalModel.prepare_timestep
PhysicalModel.getMaximumTimestep = PhysicalModel.get_maximum_timestep
PhysicalModel.setupLinearizedProblem = PhysicalModel.setup_linearized_problem
PhysicalModel.solveAdjoint = PhysicalModel.solve_adjoint
PhysicalModel.getVariableField = PhysicalModel.get_variable_field

__all__ = ["PhysicalModel"]
