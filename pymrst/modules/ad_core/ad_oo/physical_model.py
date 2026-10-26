"""MRST ``ad-core/models/PhysicalModel.m``: the AD-OO model protocol.

This is the faithful port, method for method, that MRST's model hierarchy is
built on. The legacy `pymrst.modules.ad_core.models.physical_model` keeps a
different contract - `get_equations` returns a bare list and `update_state`
takes one flat increment - that PyMRST's existing models and solver were
written against. Changing that contract would move every validated result, so
this class lives beside it, and the models ported from MRST (and
`ad_oo.NonLinearSolver`) use this one.

What MRST's contract carries that the legacy one does not:

- `get_equations` returns a `LinearizedProblem` that knows its primary
  variables, so a solved increment comes back as one block per variable (`dx`)
  rather than a vector the model has to slice by hand;
- `step_function` is one Newton iteration end to end - assemble, check
  convergence, solve, update - so the nonlinear solver only loops;
- driving forces (`W`, `bc`, `src`) are passed into every call rather than
  stored on the model, so a model is never left holding the previous control.

Property access follows MRST: `get_variable_field` names where a variable is
stored on the state, and anything it does not name is looked up in the model's
state-function groupings. Indices are 0-based; `':'` means the whole field.

State is mutable here, where MRST's is a value. Methods that return a state in
MRST still return it, but a state-function evaluation caches into the state it
was given rather than into a copy.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from pymrst.modules.ad_core.backends.auto_diff_backend import AutoDiffBackend
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.utils.adi import value

WHOLE: str = ":"


def merge_options(target: Any, options: Dict[str, Any]) -> Any:
    """MRST `merge_options`: set existing attributes, matching names
    case-insensitively, and refuse a name the target does not have."""
    names = {name.lower(): name for name in vars(target)}
    for key, item in options.items():
        name = names.get(key.lower())
        if name is None:
            raise AttributeError(
                f"{type(target).__name__} has no option {key!r}")
        setattr(target, name, item)
    return target


def merge_options_relaxed(target: Any, options: Dict[str, Any]) -> Any:
    """MRST `merge_options_relaxed`: assign each option by its exact name.

    MATLAB raises when a class has no such property; so does this, because a
    misspelled model option would otherwise be stored and never read.
    """
    for key, item in options.items():
        if not hasattr(target, key):
            raise AttributeError(
                f"{type(target).__name__} has no property {key!r}")
        setattr(target, key, item)
    return target


class StepReport(dict):
    """MRST's step report struct: a dict that also reads as attributes."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class PhysicalModel:
    """MRST `PhysicalModel`: equations, updates and convergence for a state."""

    def __init__(self, G: Any = None, **options: Any):
        self.operators: Any = None
        self.nonlinearTolerance: float = 1e-6
        self.G = None
        self.verbose = False
        self.stepFunctionIsLinear = False
        self.AutoDiffBackend = AutoDiffBackend()
        self.OutputStateFunctions: List[str] = []
        self.stateFunctionEvaluationMode = "lazy"
        self.stateFunctionGraph: Optional[Dict[str, Any]] = None
        merge_options(self, options)
        self.G = G
        self.stepFunctionIsLinear = False

    # -- primary variables and AD initialisation ----------------------------

    def get_primary_variables(self, state: State):
        """`(vars, names, origin)`. The base model has none."""
        return [], [], []

    def get_state_ad(self, state: State, init: bool = True):
        """Seed the primary variables as AD and set them on the state."""
        variables, names, origin = self.get_primary_variables(state)
        if init:
            # MRST's GenericAD carries `numVars`, one count per primary
            # variable, because the diagonal backend shares one Jacobian block
            # among variables of equal size. The counts are kept here instead.
            state["primaryVariableSizes"] = [int(np.size(value(v))) for v in variables]
            variables = list(self.AutoDiffBackend.init_variables_ad(*variables))
            state["primaryVariables"] = list(names)
        state = self.init_state_ad(state, variables, names, origin)
        if self.stateFunctionEvaluationMode.lower() == "full":
            state = self.evaluate_all_state_functions(state)
        return state, names, origin

    def get_reverse_state_ad(self, state: State, *args):
        return self.get_state_ad(state, *args)

    def init_state_ad(self, state: State, variables, names, origin) -> State:
        for name, variable in zip(names, variables):
            state = self.set_prop(state, name, variable)
        return self.init_state_function_containers(state)

    def init_state_function_containers(self, state: State) -> State:
        for grouping in self.get_state_function_groupings()[0]:
            state = grouping.init_state_function_container(state)
        return state

    # -- equations ----------------------------------------------------------

    def get_model_equations(self, state0: State, state: State, dt: float,
                            driving_forces: Any):
        raise NotImplementedError("Not implemented in base class")

    def get_equations(self, state0: State, state: State, dt: float,
                      forces: Any, *, Verbose: bool = False,
                      reverseMode: bool = False, resOnly: bool = False,
                      iteration: float = -1):
        """`(problem, state)` from `get_model_equations`."""
        state = copy_state(state)
        if reverseMode:
            primary_vars = state0["primaryVariables"]
            state, _, _ = self.get_state_ad(state, False)
        else:
            state, primary_vars, _ = self.get_state_ad(state, not resOnly)
        eqs, names, types, state = self.get_model_equations(state0, state, dt,
                                                            forces)
        problem = self.setup_linearized_problem(eqs, types, names, primary_vars,
                                                state, dt)
        return problem, state

    def get_adjoint_equations(self, state0: State, state: State, dt: float,
                              forces: Any, **options: Any):
        return self.get_equations(state0, state, dt, forces, **options)

    # -- validation ---------------------------------------------------------

    def validate_state(self, state: State) -> State:
        # MRST's states are plain structs; accept any mapping (`initResSol`
        # returns PyMRST's solver `State`) and work on an AD-OO copy.
        if not isinstance(state, State) or not hasattr(state, "containers"):
            fields = dict(state) if isinstance(state, dict) else {
                k: v for k, v in vars(state).items() if not k.startswith("_")}
            state = copy_state(fields)
        if "time" not in state:
            state["time"] = 0.0
        return state

    def validate_model(self, *args: Any) -> "PhysicalModel":
        """Set up backend operators and groupings (MRST `validateModel`).

        MRST's optional trailing `true` asks for a hash comparison between the
        grid and rock and the ones the operators were built from. MRST only
        stores those hashes when `mrstSettings('useHash')` is on (off by
        default); this port never stores them, so there is nothing to compare.
        """
        model = self.AutoDiffBackend.update_discrete_operators(self)
        model = model.setup_state_function_groupings()
        if model.stateFunctionEvaluationMode.lower() == "full":
            model, _ = model.setup_state_function_graph()
        return model

    def setup_state_function_groupings(self, set_defaults: bool = True):
        if self.verbose:
            print(f"Initializing StateFunctionGroupings attached to class of "
                  f"type {type(self).__name__}")
        return self

    def remove_state_function_groupings(self):
        self.stateFunctionGraph = None
        if self.verbose:
            print(f"Resetting StateFunctionGroupings attached to class of "
                  f"type {type(self).__name__}")
        return self

    def validate_schedule(self, schedule: Any) -> Any:
        """Check a schedule's steps and controls before anything runs.

        PyMRST's `Schedule` keeps MRST's fields under its own names
        (`step_val`, `step_control`, `controls`) with 0-based control indices.
        """
        val = np.asarray(schedule.step_val, dtype=float).ravel()
        ctrl = np.asarray(schedule.step_control, dtype=np.int64).ravel()
        if val.size != ctrl.size:
            raise ValueError(
                "The time-steps in schedule.step.val and control indices in "
                "schedule.step.control must have the same size.")
        n_control = len(schedule.controls)
        if ctrl.size and ctrl.max() >= n_control:
            raise ValueError(
                f"schedule.control has length {n_control}, but the largest "
                f"control in schedule.step.control was {int(ctrl.max()) + 1}.")
        if ctrl.size and ctrl.min() < 0:
            raise ValueError(
                "schedule.step.control must be non-negative indices into "
                "schedule.control.")
        if np.any(val < 0):
            raise ValueError("Negative time-steps present in schedule.step.val")
        for index, control in enumerate(schedule.controls):
            schedule.controls[index] = self.validate_driving_forces(control,
                                                                    index)
        return schedule

    def validate_driving_forces(self, forces: Any, index: int) -> Any:
        valid = self.get_valid_driving_forces()
        for name in _force_names(forces):
            if name not in valid:
                raise KeyError(
                    f'Provided driving force field "{name}" is not present in '
                    f"the output from model.getValidDrivingForces().")
        return forces

    # -- timestep hooks -----------------------------------------------------

    def prepare_report_step(self, state: State, state0: State, dT: float,
                            driving_forces: Any):
        model = self.validate_model(driving_forces)
        return model, state

    def prepare_timestep(self, state: State, state0: State, dt: float,
                         driving_forces: Any):
        return self, state

    def update_state(self, state: State, problem: Any, dx: Sequence[Any],
                     forces: Any):
        for name in problem.primary_variables:
            state, _, _ = self.update_state_from_increment(state, dx, problem,
                                                           name)
        return state, None

    def get_maximum_timestep(self, state: State, state0: State, dt: float,
                             driving_forces: Any) -> float:
        return float("inf")

    def update_for_changed_controls(self, state: State, forces: Any):
        return self, state

    def reduce_state(self, state: State, remove_containers: bool = True) -> State:
        """Drop the AD and, by default, the state-function caches.

        Kept caches are reduced too (MRST's `value(state)` reaches into the
        containers): a converged state's `ComponentTotalMass` must be numbers,
        or the next step's `mass - mass0` cancels the accumulation derivative.
        """
        kept = {} if remove_containers else {
            name: ValueContainer({fn: _reduced_cached_value(item)
                                  for fn, item in container.items()})
            for name, container in state.containers.items()}
        reduced = copy_state(State({key: _reduced_value(item)
                                    for key, item in state.items()
                                    if key not in ("primaryVariables",
                                                   "primaryVariableSizes",
                                                   "evaluated")}))
        reduced.containers = kept
        return reduced

    def update_after_convergence(self, state0: State, state: State, dt: float,
                                 driving_forces: Any):
        """Keep only the state functions named in `OutputStateFunctions`."""
        for grouping in self.get_state_function_groupings()[0]:
            struct_name = grouping.get_state_function_container_name()
            if struct_name not in state.containers:
                continue
            names = grouping.get_names()
            current = [n for n in names if n in self.OutputStateFunctions]
            if current:
                state.containers[struct_name] = ValueContainer({
                    name: self.get_prop(state, name) for name in current})
            else:
                del state.containers[struct_name]
        return state, StepReport()

    # -- convergence --------------------------------------------------------

    def check_convergence(self, problem: Any, *args: Any):
        values, tolerances, names = self.get_convergence_values(problem, *args)
        return np.asarray(values) < np.asarray(tolerances), values, names

    def get_convergence_values(self, problem: Any, n: float = np.inf):
        values = problem.norm(n)
        tolerances = np.full(np.shape(values), self.nonlinearTolerance)
        names = [f"{name} ({kind})" for name, kind
                 in zip(problem.equation_names, problem.types)]
        return values, tolerances, names

    def step_function(self, state: State, state0: State, dt: float,
                      driving_forces: Any, linsolver: Any, nonlinsolver: Any,
                      iteration: int, **options: Any):
        """One Newton iteration: assemble, check, solve, update."""
        out_of_iterations = iteration > nonlinsolver.maxIterations
        timer = perf_counter()
        problem, state = self.get_equations(state0, state, dt, driving_forces,
                                            resOnly=out_of_iterations,
                                            iteration=iteration, **options)
        problem.iteration_no = iteration
        problem.driving_forces = driving_forces
        t_assembly = perf_counter() - timer

        convergence, values, resnames = self.check_convergence(problem)
        done_min_its = iteration > nonlinsolver.minIterations
        failure_msg = ""
        failure = False
        linear_report: Any = {}
        update_report: Any = {}
        stabilize_report: Any = {}
        if not (np.all(convergence) and done_min_its) and not out_of_iterations:
            try:
                dx, _, linear_report = linsolver.solve_linear_problem(problem, self)
                if any(not np.all(np.isfinite(value(d))) for d in dx):
                    failure = True
                    failure_msg = "Linear solver produced non-finite values."
            except (ArithmeticError, RuntimeError, ValueError,
                    np.linalg.LinAlgError) as error:
                failure = True
                failure_msg = f"Linear solver failure: {type(error).__name__}: {error}"
            if not failure:
                dx, stabilize_report = nonlinsolver.stabilize_newton_increments(
                    self, problem, dx)
                state = self.reduce_state(state, True)
                is_struggling = (nonlinsolver.alwaysUseStabilization
                                 or nonlinsolver.convergenceIssues)
                if is_struggling and nonlinsolver.useLinesearch:
                    state, update_report, linesearch = nonlinsolver.apply_linesearch(
                        self, state0, state, problem, dx, driving_forces, **options)
                    stabilize_report = dict(stabilize_report or {},
                                            linesearch=linesearch)
                else:
                    state, update_report = self.update_state(state, problem, dx,
                                                             driving_forces)
        else:
            state = self.reduce_state(state, False)

        model_converged = bool(np.all(convergence))
        if out_of_iterations and nonlinsolver.acceptanceFactor != 1:
            values, tol, resnames = self.get_convergence_values(problem)
            model_converged = bool(np.all(
                np.asarray(values) < nonlinsolver.acceptanceFactor*np.asarray(tol)))
        is_converged = (model_converged and done_min_its) or self.stepFunctionIsLinear
        if self.stepFunctionIsLinear:
            _, state = self.get_equations(state0, state, dt, driving_forces,
                                          resOnly=True, iteration=iteration + 1,
                                          **options)
            state = self.reduce_state(state, True)
        if self.verbose:
            from pymrst.modules.ad_core.utils.print_convergence_report import (
                print_convergence_report,
            )
            print_convergence_report(resnames, values, convergence, iteration, is_converged)
        report = self.make_step_report(
            LinearSolver=linear_report, UpdateState=update_report,
            Failure=failure, FailureMsg=failure_msg, Converged=is_converged,
            Solved=not is_converged, AssemblyTime=t_assembly,
            Residuals=np.asarray(values), StabilizeReport=stabilize_report,
            ResidualsConverged=np.asarray(convergence))
        return state, report

    def make_step_report(self, **fields: Any) -> StepReport:
        report = StepReport(LinearSolver=None, UpdateState=None, Failure=False,
                            FailureMsg="", Converged=False, Solved=True,
                            FinalUpdate=None, Residuals=None, AssemblyTime=0.0,
                            StabilizeReport=None, ResidualsConverged=None)
        for key, item in fields.items():
            if key not in report:
                raise KeyError(f"unknown step report field {key!r}")
            report[key] = item
        return report

    def solve_adjoint(self, solver: Any, get_state, get_obj, schedule: Any,
                      lam: Any, step_no: int, **options: Any):
        """One backward step of the adjoint sweep (1-based `step_no`)."""
        dt_steps = np.asarray(schedule.step_val, dtype=float)
        current = get_state(step_no)
        before = get_state(step_no - 1)
        dt = dt_steps[step_no - 1]
        control_of = lambda step: schedule.controls[int(schedule.step_control[step - 1])]
        forces = self._merged_forces(control_of(step_no))
        model = self._validated_for(forces)
        if step_no == 1:
            before = model.validate_state(before)
        problem, _ = model.get_adjoint_equations(before, current, dt, forces,
                                                 reverseMode=False,
                                                 iteration=float("inf"))
        if step_no < dt_steps.size:
            after = get_state(step_no + 1)
            dt_next = dt_steps[step_no]
            current, primary_vars, _ = model.get_reverse_state_ad(current)
            assert list(current["primaryVariables"]) == list(primary_vars)
            if schedule.step_control[step_no] == schedule.step_control[step_no - 1]:
                forces_p = forces
            else:
                forces_p = self._merged_forces(control_of(step_no + 1))
                model = self._validated_for(forces_p)
            problem_p, _ = model.get_adjoint_equations(
                current, after, dt_next, forces_p, iteration=float("inf"),
                reverseMode=True)
        else:
            problem_p = None
        lam, lambda_vec, rep = solver.solve_adjoint_problem(
            problem_p, problem, lam, get_obj(step_no, model, problem.state),
            model, **options)
        report = {"Types": problem.types, "LinearSolverReport": rep}
        return lam, lambda_vec, report

    def _merged_forces(self, control: Any) -> Dict[str, Any]:
        forces = dict(self.get_valid_driving_forces())
        forces.update(self.get_driving_forces(control))
        return forces

    def _validated_for(self, forces: Dict[str, Any]) -> "PhysicalModel":
        facility = getattr(self, "FacilityModel", None)
        if facility is not None and hasattr(self, "fluid"):
            self.FacilityModel = facility.validate_model(forces)
            return self
        return self.validate_model(forces)

    # -- property access ----------------------------------------------------

    def get_variable_field(self, name: str, throw_error: bool = True):
        """`(field, index)` for a stored variable. The base knows none."""
        if throw_error:
            raise KeyError(
                f"PhysicalModel:UnknownVariable: State variable '{name}' is "
                f"not known to this model")
        return None, None

    def get_prop(self, state: State, name: str) -> Any:
        """A stored variable, or a state function from a grouping."""
        field, index = self.get_variable_field(name, False)
        if field is None:
            groupings = self.get_state_function_groupings()[0]
            if groupings and not groupings[0].is_state_initialized(state):
                # MRST's `getProp` initialises the containers on its own copy
                # of the state: a caller that does not take the returned state
                # back does not get the caches. Initialising in place would
                # carry them into, e.g., the next step's start state.
                state = self.init_state_function_containers(copy_state(state))
            elif any(isinstance(c, ValueContainer) for c in state.containers.values()):
                # Value caches (MRST's plain structs) are written on the copy
                # only; handle caches stay shared, as `copy_state` keeps them.
                state = copy_state(state)
            for grouping in groupings:
                for function_name in grouping.get_names():
                    if function_name.lower() == name.lower():
                        return grouping.get(self, state, function_name)
            raise KeyError(
                f"PhysicalModel:UnknownVariable: I did not find {name} in any "
                f"state function grouping or via getVariableField for model "
                f"of type {type(self).__name__}")
        stored = _field_get(state, field)
        if isinstance(stored, list):
            return stored if index == WHOLE else stored[index]
        if index == WHOLE or _column_count(stored) == 1:
            return stored
        return np.asarray(stored)[:, index]

    def get_props(self, state: State, *names: str) -> Any:
        out = [self.get_prop(state, name) for name in names]
        return out[0] if len(out) == 1 else out

    def increment_prop(self, state: State, name: str, increment: Any) -> State:
        field, index = self.get_variable_field(name)
        current = _field_get(state, field)
        if index == WHOLE or _column_count(current) == 1:
            _field_set(state, field, current + increment)
        else:
            updated = np.array(current, dtype=float, copy=True)
            updated[:, index] = updated[:, index] + increment
            _field_set(state, field, updated)
        return state

    def set_prop(self, state: State, name: str, new_value: Any) -> State:
        """Write a variable; an AD column turns its field into a column list.

        MRST's own rule: a field that is one column, or written whole, is
        replaced. Writing an AD value into one column of a numeric matrix
        converts the matrix to a list of columns first, because a numeric
        array cannot hold derivatives.
        """
        field, index = self.get_variable_field(name)
        unit = _field_has(state, field) and _column_count(_field_get(state, field)) == 1
        if index == WHOLE or unit:
            _field_set(state, field, new_value)
            return state
        stored = _field_get(state, field)
        if hasattr(new_value, "jac") and not isinstance(stored, list):
            stored = [np.asarray(stored)[:, column]
                      for column in range(np.shape(stored)[1])]
        if isinstance(stored, list):
            stored = list(stored)
            stored[index] = new_value
        else:
            stored = np.array(stored, dtype=float, copy=True)
            stored[:, index] = new_value
        _field_set(state, field, stored)
        return state

    def set_props(self, state: State, names: Sequence[str],
                  values: Sequence[Any]) -> State:
        assert len(names) == len(values)
        for name, item in zip(names, values):
            state = self.set_prop(state, name, item)
        return state

    def get_increment(self, dx: Sequence[Any], problem: Any, name: str) -> Any:
        """MRST: `isVar = problem.indexOfPrimaryVariable(name)`, then `dx{isVar}`
        if any, else 0. A position (or None) is accepted as well."""
        found = problem.index_of_primary_variable(name)
        if isinstance(found, np.ndarray):
            hits = np.flatnonzero(found)
            return dx[int(hits[0])] if hits.size else 0
        return 0 if found is None else dx[found]

    def update_state_from_increment(self, state: State, dx: Any, problem: Any,
                                    name: str, relchangemax: Any = np.inf,
                                    abschangemax: Any = np.inf):
        """Apply one variable's increment, scaled to respect the limits.

        `dx` is either the full increment list or this variable's increment.
        The limits scale whole rows uniformly, so a multi-column update (a
        saturation row) keeps its direction and only gets shorter.
        """
        dv = (self.get_increment(dx, problem, name) if isinstance(dx, list)
              else dx)
        val0 = self.get_prop(state, name)
        has_rel = np.size(relchangemax) > 1 or np.isfinite(np.ravel(relchangemax)[0])
        has_abs = np.size(abschangemax) > 1 or np.isfinite(np.ravel(abschangemax)[0])
        if not (has_rel or has_abs):
            val = val0 + dv
        else:
            dv2 = np.asarray(value(dv), dtype=float)
            flat = dv2.ndim < 2
            dv2 = dv2.reshape(dv2.shape[0], -1) if dv2.ndim else dv2.reshape(1, 1)
            base = np.asarray(value(val0), dtype=float).reshape(dv2.shape[0], -1)
            change = np.full(dv2.shape[0], np.inf)
            with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                if has_rel:
                    biggest = np.max(np.abs(dv2/base), axis=1)
                    change = np.minimum(change, np.minimum(
                        np.ravel(relchangemax)/biggest, 1.0))
                if has_abs:
                    biggest = np.max(np.abs(dv2), axis=1)
                    change = np.minimum(change, np.minimum(
                        np.ravel(abschangemax)/biggest, 1.0))
            # MATLAB's min ignores the NaN of 0/0; a row that did not move
            # takes the full (zero) step.
            change = np.where(np.isnan(change), 1.0, change)
            scaled = dv2*change[:, None]
            val = val0 + (scaled.ravel() if flat else scaled)
        state = self.set_prop(state, name, val)
        return state, val, val0

    def cap_property(self, state: State, name: str, minvalue: Any,
                     maxvalue: Any = None) -> State:
        current = self.get_prop(state, name)
        capped = _ad_maximum(current, minvalue)
        if maxvalue is not None:
            capped = _ad_minimum(capped, maxvalue)
        return self.set_prop(state, name, capped)

    def get_driving_forces(self, control: Any) -> Dict[str, Any]:
        """A control's fields as a mapping (MRST returns name-value pairs)."""
        if control is None:
            return {}
        if isinstance(control, dict):
            return dict(control)
        return {k: v for k, v in vars(control).items() if not k.startswith("_")}

    def get_valid_driving_forces(self) -> Dict[str, Any]:
        return {"stopFunction": lambda model, state, state_prev: False,
                "extra": None}

    def check_property(self, state: State, prop: str, n_el: Any, dim: Any) -> None:
        if np.size(dim) > 1:
            assert np.size(n_el) == np.size(dim)
            for count, axis in zip(np.ravel(n_el), np.ravel(dim)):
                self.check_property(state, prop, int(count), int(axis))
            return
        field, _ = self.get_variable_field(prop)
        assert _field_has(state, field), (
            f'Field ".{field}" missing! {prop} must be supplied for model '
            f'"{type(self).__name__}"')
        stored = _field_get(state, field)
        if isinstance(stored, list):
            shape = (np.size(value(stored[0])), len(stored))
        else:
            array = np.asarray(value(stored))
            shape = array.shape if array.ndim > 1 else (array.size, 1)
        n_actual = shape[int(dim) - 1] if int(dim) <= len(shape) else 1
        which = {1: "rows", 2: "columns"}.get(int(dim), f"dimension {dim}")
        assert n_actual == n_el, (
            f'Dimension mismatch for {which} of property "{prop}" '
            f"(state.{field}): Expected {which} to have {n_el} entries but "
            f"state had {n_actual} instead.")

    def setup_linearized_problem(self, eqs, types, names, primary_vars,
                                 state: State, dt: float):
        from pymrst.modules.ad_core.solvers.linear_solver_ad import LinearizedProblem
        return LinearizedProblem.from_equations(eqs, types, names, primary_vars,
                                                state, dt)

    # -- state-function groupings -------------------------------------------

    def set_state_function_evaluation_mode(self, mode: str) -> "PhysicalModel":
        if mode.lower() not in ("lazy", "full"):
            raise ValueError(
                f"Unknown mode '{mode}'. Possible options are:\n"
                f"  - 'lazy' for evaluation when needed\n"
                f"  - 'full' for pre-evaluation of all properties.")
        if self.verbose:
            print(f'Setting evaluation mode to "{mode}"')
        self.stateFunctionEvaluationMode = mode
        return self

    def get_state_function_evaluation_mode(self) -> str:
        return self.stateFunctionEvaluationMode

    def setup_state_function_graph(self, graph: Optional[Dict[str, Any]] = None,
                                   filter: Sequence[str] = (),
                                   filter_origin: Any = "all"):
        """Order every state function so each comes after its dependencies.

        MRST builds the graph across all groupings and takes a topological
        order with matlab_bgl; this walks the same dependencies depth first.
        `filter` entries are `"Group.name"`, `"Group.*"` or a bare `"name"`.
        """
        rebuild = graph is None and self.stateFunctionGraph is None
        if graph is None:
            graph = self.stateFunctionGraph
        if rebuild or filter or filter_origin != "all":
            groups, names, origin = self.get_state_function_groupings()
            keep = _origin_filter(self, origin, filter_origin)
            groups = [g for g, k in zip(groups, keep) if k]
            names = [n for n, k in zip(names, keep) if k]
            graph = _dependency_graph(groups, names)
        order = list(graph["EvaluationOrder"])
        skip = set()
        for entry in filter:
            parts = entry.split(".")
            for group_name, function_name in order:
                if len(parts) == 2:
                    if parts[0].lower() == group_name.lower() and (
                            parts[1] == "*" or parts[1].lower() == function_name.lower()):
                        skip.add((group_name, function_name))
                elif parts[0].lower() == function_name.lower():
                    skip.add((group_name, function_name))
        graph = dict(graph)
        graph["EvaluationOrder"] = [p for p in order if p not in skip]
        self.stateFunctionGraph = graph
        return self, graph

    def evaluate_all_state_functions(self, state: State) -> State:
        groups, names, models = self.get_state_function_groupings()
        assert self.stateFunctionGraph is not None
        by_name = dict(zip(names, zip(groups, models)))
        state["evaluated"] = True
        for group_name, function_name in self.stateFunctionGraph["EvaluationOrder"]:
            group, model = by_name[group_name]
            state = group.evaluate_state_function_unsafe(model, state,
                                                         function_name)
        return state

    def get_state_function_groupings(self):
        """`(groupings, names, models)`. The base model owns none."""
        return [], [], []

    def check_state_function_dependencies(self) -> None:
        groups = self.get_state_function_groupings()[0]
        for group in groups:
            group.check_dependencies(groups)

    # -- static update limits -----------------------------------------------

    @staticmethod
    def limit_update_relative(dv: Any, val: Any, max_rel_ch: float):
        with np.errstate(divide="ignore", invalid="ignore"):
            biggest = np.max(np.abs(np.atleast_2d(dv.T).T/np.atleast_2d(val.T).T),
                             axis=1)
            change = np.minimum(max_rel_ch/biggest, 1.0)
        return dv*change.reshape(-1, *([1]*(np.ndim(dv) - 1))), change

    @staticmethod
    def limit_update_absolute(dv: Any, max_abs_ch: float):
        with np.errstate(divide="ignore", invalid="ignore"):
            biggest = np.max(np.abs(np.atleast_2d(dv.T).T), axis=1)
            change = np.minimum(max_abs_ch/biggest, 1.0)
        return dv*change.reshape(-1, *([1]*(np.ndim(dv) - 1))), change

    @staticmethod
    def strip_vars(variables: Sequence[str], names: Any):
        """Drop named variables (case-insensitive); `(kept, is_removed)`."""
        names = [names] if isinstance(names, str) else list(names)
        lowered = [n.lower() for n in names]
        is_removed = [v.lower() in lowered for v in variables]
        kept = [v for v, gone in zip(variables, is_removed) if not gone]
        return kept, is_removed

    # MRST spellings
    getPrimaryVariables = get_primary_variables
    getStateAD = get_state_ad
    getReverseStateAD = get_reverse_state_ad
    initStateAD = init_state_ad
    initStateFunctionContainers = init_state_function_containers
    getModelEquations = get_model_equations
    getEquations = get_equations
    getAdjointEquations = get_adjoint_equations
    validateState = validate_state
    validateModel = validate_model
    setupStateFunctionGroupings = setup_state_function_groupings
    removeStateFunctionGroupings = remove_state_function_groupings
    validateSchedule = validate_schedule
    validateDrivingForces = validate_driving_forces
    prepareReportstep = prepare_report_step
    prepareTimestep = prepare_timestep
    updateState = update_state
    getMaximumTimestep = get_maximum_timestep
    updateForChangedControls = update_for_changed_controls
    reduceState = reduce_state
    updateAfterConvergence = update_after_convergence
    checkConvergence = check_convergence
    getConvergenceValues = get_convergence_values
    stepFunction = step_function
    makeStepReport = make_step_report
    solveAdjoint = solve_adjoint
    getVariableField = get_variable_field
    getProp = get_prop
    getProps = get_props
    incrementProp = increment_prop
    setProp = set_prop
    setProps = set_props
    getIncrement = get_increment
    updateStateFromIncrement = update_state_from_increment
    capProperty = cap_property
    getDrivingForces = get_driving_forces
    getValidDrivingForces = get_valid_driving_forces
    checkProperty = check_property
    setupLinearizedProblem = setup_linearized_problem
    setStateFunctionEvaluationMode = set_state_function_evaluation_mode
    getStateFunctionEvaluationMode = get_state_function_evaluation_mode
    setupStateFunctionGraph = setup_state_function_graph
    evaluateAllStateFunctions = evaluate_all_state_functions
    getStateFunctionGroupings = get_state_function_groupings
    checkStateFunctionDependencies = check_state_function_dependencies
    limitUpdateRelative = limit_update_relative
    limitUpdateAbsolute = limit_update_absolute
    stripVars = strip_vars


class ValueContainer(dict):
    """A state-function cache with MATLAB value semantics.

    MRST keeps its caches in `HandleStruct` handles while equations are
    assembled, so a copy of the state (such as a flow state) shares them and
    what is written into one is seen through the other; a plain dict here.
    `reduceState(state, false)` turns them into plain structs of values
    (`HandleStruct.value`), as does `updateAfterConvergence` for the kept
    outputs: those are copied with the state, and what `getProp` evaluates
    into them stays on its own copy. This marks such a cache.
    """


def copy_state(state: Any) -> State:
    """MATLAB's value semantics for a state: a new struct, new well rows.

    Fields are replaced rather than written into throughout this port, so a
    shallow copy of the state is enough - except `wellSol`, whose rows are
    records the well models update attribute by attribute.
    """
    from copy import copy
    out = State(dict(state))
    out.containers = {name: ValueContainer(c) if isinstance(c, ValueContainer) else c
                      for name, c in getattr(state, "containers", {}).items()}
    if "wellSol" in out and out["wellSol"] is not None:
        out["wellSol"] = [copy(row) for row in out["wellSol"]]
    return out


def _force_names(forces: Any) -> List[str]:
    if forces is None:
        return []
    if isinstance(forces, dict):
        return list(forces)
    return [k for k in vars(forces) if not k.startswith("_")]


def _field_get(container: Any, name: str) -> Any:
    """A struct field: dict item (a state) or attribute (a wellSol row)."""
    if isinstance(container, dict):
        return container[name]
    return getattr(container, name)


def _field_set(container: Any, name: str, item: Any) -> None:
    if isinstance(container, dict):
        container[name] = item
    else:
        setattr(container, name, item)


def _field_has(container: Any, name: str) -> bool:
    if isinstance(container, dict):
        return name in container
    return hasattr(container, name)


def _column_count(item: Any) -> int:
    """Columns in a stored field: an AD vector or a 1-D array is one."""
    if isinstance(item, list):
        return len(item)
    if hasattr(item, "jac") or np.ndim(item) < 2:
        return 1
    return int(np.shape(item)[1])


def _reduced_value(item: Any) -> Any:
    """MRST `value(state)`: AD to numbers, a column list back to a matrix."""
    if isinstance(item, list) and item and all(
            hasattr(x, "jac") or isinstance(x, np.ndarray) for x in item):
        columns = [np.asarray(value(x), dtype=float).ravel() for x in item]
        if len({c.size for c in columns}) == 1:
            return np.column_stack(columns)
    if hasattr(item, "jac"):
        return np.asarray(value(item), dtype=float)
    return item


def _reduced_cached_value(item: Any) -> Any:
    """A cached state-function value without AD, keeping its list layout
    (MRST flattens only row cells; Python lists carry no orientation)."""
    if isinstance(item, list):
        return [_reduced_cached_value(x) for x in item]
    if hasattr(item, "jac"):
        return np.asarray(value(item), dtype=float)
    return item


def _ad_maximum(current: Any, bound: Any) -> Any:
    """MATLAB `max(bound, v)` that keeps AD where `v` is the larger."""
    if hasattr(current, "jac"):
        # A fresh AD object: `__setitem__` writes in place, and the caller's
        # value must not change underneath it.
        out = current + 0.0
        low = np.asarray(value(current)) < bound
        if np.any(low):
            out[np.flatnonzero(low)] = np.broadcast_to(bound, low.shape)[low]
        return out
    return np.maximum(bound, current)


def _ad_minimum(current: Any, bound: Any) -> Any:
    if hasattr(current, "jac"):
        out = current + 0.0
        high = np.asarray(value(current)) > bound
        if np.any(high):
            out[np.flatnonzero(high)] = np.broadcast_to(bound, high.shape)[high]
        return out
    return np.minimum(current, bound)


def _origin_filter(model: Any, origin: Sequence[Any], selector: Any) -> List[bool]:
    if isinstance(selector, (list, tuple)):
        return [any(isinstance(o, cls) for cls in selector) for o in origin]
    if selector == "all":
        return [True]*len(origin)
    if selector == "self":
        return [type(o) is type(model) for o in origin]
    raise ValueError(f"No such origin filter {selector}")


def _dependency_graph(groups: Sequence[Any], names: Sequence[str]) -> Dict[str, Any]:
    """Topological order of `(grouping, function)` pairs across groupings."""
    by_struct = {g.struct_name: n for g, n in zip(groups, names)}
    owner = dict(zip(names, groups))
    order: List[Tuple[str, str]] = []
    seen: set = set()

    def visit(node: Tuple[str, str], stack: Tuple[Tuple[str, str], ...]) -> None:
        if node in seen:
            return
        if node in stack:
            raise ValueError(f"cyclic state-function dependency at {node}")
        group = owner[node[0]]
        for dep in group.get_state_function(node[1]).dependencies:
            if dep.grouping == "state":
                continue
            target = node[0] if dep.grouping == "self" else by_struct.get(
                dep.grouping, dep.grouping)
            if target in owner:
                visit((target, dep.name), stack + (node,))
        seen.add(node)
        order.append(node)

    for name, group in zip(names, groups):
        for function_name in group.get_names():
            visit((name, function_name), ())
    return {"EvaluationOrder": order, "GroupNames": list(names)}


__all__ = ["PhysicalModel", "StepReport", "copy_state", "merge_options",
           "merge_options_relaxed"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.utils.adi import init_variables_adi

    class _Model(PhysicalModel):
        """Two stored variables: a scalar field and a three-column one."""

        def get_variable_field(self, name, throw_error=True):
            fields = {"p": ("pressure", WHOLE), "s": ("s", WHOLE),
                      "sw": ("s", 0), "so": ("s", 1), "sg": ("s", 2)}
            if name.lower() in fields:
                return fields[name.lower()]
            return super().get_variable_field(name, throw_error)

    def _state():
        return State(pressure=np.array([1.0, 2.0]),
                     s=np.array([[0.2, 0.7, 0.1], [0.3, 0.3, 0.4]]))

    def test_options_match_case_insensitively_and_refuse_unknowns():
        model = PhysicalModel(None, nonlineartolerance=1e-3)
        assert model.nonlinearTolerance == 1e-3
        try:
            PhysicalModel(None, noSuchOption=1)
            raise SystemExit("should have raised")
        except AttributeError:
            pass
        print("ok test_options_match_case_insensitively_and_refuse_unknowns")

    def test_get_prop_reads_a_column_and_set_prop_writes_one():
        model, state = _Model(), _state()
        np.testing.assert_allclose(model.get_prop(state, "so"), [0.7, 0.3])
        model.set_prop(state, "sg", np.array([0.0, 0.5]))
        np.testing.assert_allclose(state["s"][:, 2], [0.0, 0.5])
        print("ok test_get_prop_reads_a_column_and_set_prop_writes_one")

    def test_setting_an_ad_column_turns_the_field_into_columns():
        model, state = _Model(), _state()
        (sw,) = init_variables_adi(np.array([0.25, 0.35]))
        model.set_prop(state, "sw", sw)
        assert isinstance(state["s"], list) and hasattr(state["s"][0], "jac")
        np.testing.assert_allclose(model.get_prop(state, "sg"), [0.1, 0.4])
        reduced = model.reduce_state(state)
        np.testing.assert_allclose(reduced["s"][:, 0], [0.25, 0.35])
        print("ok test_setting_an_ad_column_turns_the_field_into_columns")

    def test_increment_limits_scale_whole_rows():
        model, state = _Model(), _state()
        problem = SimpleNamespace(index_of_primary_variable=lambda n: None)
        ds = np.array([[0.4, -0.4, 0.0], [0.1, -0.1, 0.0]])
        model.update_state_from_increment(state, ds, problem, "s", np.inf, 0.2)
        # Row one is halved to respect the 0.2 limit; row two is untouched.
        np.testing.assert_allclose(state["s"], [[0.4, 0.5, 0.1], [0.4, 0.2, 0.4]])
        print("ok test_increment_limits_scale_whole_rows")

    def test_relative_limit_on_pressure():
        model, state = _Model(), _state()
        problem = SimpleNamespace(index_of_primary_variable=lambda n: 0)
        model.update_state_from_increment(state, [np.array([5.0, 0.1])],
                                          problem, "p", 0.5)
        np.testing.assert_allclose(state["pressure"], [1.5, 2.1])
        print("ok test_relative_limit_on_pressure")

    def test_cap_property_keeps_derivatives_inside_the_bounds():
        model, state = _Model(), _state()
        (p,) = init_variables_adi(np.array([-1.0, 2.0]))
        state["pressure"] = p
        model.cap_property(state, "p", 0.0, 1.5)
        np.testing.assert_allclose(value(state["pressure"]), [0.0, 1.5])
        print("ok test_cap_property_keeps_derivatives_inside_the_bounds")

    def test_strip_vars_is_case_insensitive():
        kept, removed = PhysicalModel.strip_vars(["pressure", "sW", "x"], ["sw", "X"])
        assert kept == ["pressure"] and removed == [False, True, True]
        print("ok test_strip_vars_is_case_insensitive")

    def test_unknown_property_is_an_error_not_a_none():
        try:
            _Model().get_prop(_state(), "Nothing")
            raise SystemExit("should have raised")
        except KeyError:
            pass
        print("ok test_unknown_property_is_an_error_not_a_none")

    def test_check_property_reports_the_wrong_dimension():
        model, state = _Model(), _state()
        model.check_property(state, "s", [2, 3], [1, 2])
        try:
            model.check_property(state, "s", 4, 2)
            raise SystemExit("should have raised")
        except AssertionError as error:
            assert "columns" in str(error)
        print("ok test_check_property_reports_the_wrong_dimension")

    test_options_match_case_insensitively_and_refuse_unknowns()
    test_get_prop_reads_a_column_and_set_prop_writes_one()
    test_setting_an_ad_column_turns_the_field_into_columns()
    test_increment_limits_scale_whole_rows()
    test_relative_limit_on_pressure()
    test_cap_property_keeps_derivatives_inside_the_bounds()
    test_strip_vars_is_case_insensitive()
    test_unknown_property_is_an_error_not_a_none()
    test_check_property_reports_the_wrong_dimension()
    print("\nAll tests passed!")
