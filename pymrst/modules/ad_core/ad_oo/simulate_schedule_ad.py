"""MRST ``ad-core/simulators/simulateScheduleAD.m``.

Run an AD-OO model through a schedule: validate the model, schedule and
initial state once, then solve each report step with the nonlinear solver,
switching the model's driving forces whenever the step's control changes.
Returns `(wellSols, states, schedulereport)` as MRST does.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any, Callable, Dict, List, Optional, Sequence
import warnings

import numpy as np

from pymrst.modules.ad_core.ad_oo.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.ad_oo.physical_model import PhysicalModel
from pymrst.utils.format_time_range import format_time_range
from pymrst.utils.units import DAY


def simulate_schedule_ad(initState: Any, model: Any, schedule: Any, *,
                         Verbose: bool = False, OutputMinisteps: bool = False,
                         initialGuess: Optional[Sequence[Any]] = None,
                         NonLinearSolver_: Optional[NonLinearSolver] = None,
                         OutputHandler: Any = None, WellOutputHandler: Any = None,
                         ReportHandler: Any = None,
                         afterStepFn: Optional[Callable] = None,
                         controlLogicFn: Optional[Callable] = None,
                         processOutputFn: Optional[Callable] = None,
                         restartStep: int = 1, outputOffset: Optional[int] = None,
                         LinearSolver: Any = None, checkOperators: bool = False,
                         **options: Any):
    """`(wellSols, states, schedulereport)` over every report step.

    MRST's `'NonLinearSolver'` option is also accepted under that name
    through `**options`; it is spelled `NonLinearSolver_` in the signature
    only because the class of the same name is imported here.
    """
    assert isinstance(model, PhysicalModel), (
        "The model must be derived from PhysicalModel")
    solver = options.pop("NonLinearSolver", NonLinearSolver_)
    if options:
        raise TypeError(f"unknown options {sorted(options)}")
    step_val = np.asarray(schedule.step_val, dtype=float)
    tm = np.concatenate([[0.0], np.cumsum(step_val)])
    if restartStep != 1:
        T0 = float(step_val[:restartStep - 1].sum())
        if "time" in initState and abs(initState["time"] - T0) > \
                10*np.spacing(step_val.sum()):
            warnings.warn(f"Time mismatch in initial state for restart. "
                          f"Expected {format_time_range(T0)}, got "
                          f"{format_time_range(initState['time'])}.")
        n_step = step_val.size
        assert 1 < restartStep <= n_step, (
            f"Restart step must be an index between 1 and {n_step}.")
        from pymrst.modules.ad_core.utils.schedule import Schedule
        schedule = Schedule(step_val=step_val[restartStep - 1:],
                            step_control=np.asarray(schedule.step_control)[restartStep - 1:],
                            controls=list(schedule.controls))
        step_val = np.asarray(schedule.step_val, dtype=float)
    if Verbose:
        _simulation_header(tm.size - 1, tm[-1])
    if solver is None:
        solver = NonLinearSolver(LinearSolver=LinearSolver)
    else:
        assert isinstance(solver, NonLinearSolver), (
            "Passed nonlinear solver is not an instance of NonLinearSolver class!")
        if LinearSolver is not None:
            solver.LinearSolver = LinearSolver
    solver.timeStepSelector.reset()
    n_steps = step_val.size
    well_sols: List[Any] = [None]*n_steps
    states: List[Any] = [None]*n_steps
    reports: List[Any] = [None]*n_steps
    if Verbose:
        print("Validating model...")
    control_index = np.asarray(schedule.step_control, dtype=np.int64)
    forces = model.get_driving_forces(schedule.controls[control_index[0]])
    # MATLAB passes the model by value: validating it (and setting up its
    # wells) must not set up the caller's.
    from pymrst.modules.ad_core.ad_oo.wrapper_model import value_copy_model
    model = value_copy_model(model).validate_model(forces, checkOperators)
    if Verbose:
        print("Model has been validated.\nChecking state functions and dependencies...")
    model.check_state_function_dependencies()
    if Verbose:
        print("All checks ok. Model ready for simulation.\nPreparing schedule for simulation...")
    schedule = model.validate_schedule(schedule)
    if Verbose:
        print("All steps ok. Schedule ready for simulation.\nValidating initial state...")
    state = model.validate_state(initState)
    if Verbose:
        print("Initial state ok. Ready to begin simulation.")
    failure = False
    simtime = np.zeros(n_steps)
    prev_control = None
    first_empty = 0
    report = {"EarlyStop": False}
    i = -1
    for i in range(n_steps):
        _step_header(Verbose, tm, restartStep, i)
        state0 = state
        curr_control = int(control_index[i])
        if prev_control != curr_control:
            forces = model.get_driving_forces(schedule.controls[curr_control])
            model, state0 = model.update_for_changed_controls(state, forces)
            prev_control = curr_control
        extra: Dict[str, Any] = {}
        if initialGuess:
            extra["initialGuess"] = model.validate_state(initialGuess[i])
        dt = float(step_val[i])
        timer = perf_counter()
        state, report, substates = solver.solve_timestep(
            state0, dt, model, **forces, controlId=curr_control, **extra)
        if OutputMinisteps:
            ind = list(range(first_empty, first_empty + len(substates)))
            first_empty += len(substates)
        else:
            substates = [state]
            ind = [i]
        simtime[i] = perf_counter() - timer
        if not report["Converged"]:
            warnings.warn("Nonlinear solver aborted, returning incomplete results")
            failure = True
            break
        if controlLogicFn is not None:
            state, schedule, report, is_altered = controlLogicFn(state, schedule,
                                                                 report, i)
            if is_altered:
                prev_control = None
                substates[-1] = state
        if Verbose:
            _disp_step_convergence(report["Iterations"], simtime[i])
        well_sols_step = [s.get("wellSol") for s in substates]
        if processOutputFn is not None:
            substates, well_sols_step, report = processOutputFn(substates,
                                                                well_sols_step,
                                                                report)
        _write_output(OutputHandler, outputOffset, restartStep, ind, substates)
        _write_output(WellOutputHandler, outputOffset, restartStep, ind,
                      well_sols_step)
        _write_output(ReportHandler, None, restartStep, [i], [report])
        _place(well_sols, ind, well_sols_step)
        _place(states, ind, substates)
        reports[i] = report
        if afterStepFn is not None:
            model, states, reports, solver, ok = afterStepFn(model, states, reports,
                                                            solver, schedule,
                                                            simtime)
            if not ok:
                warnings.warn("Aborting due to external function")
                break
        if report["EarlyStop"]:
            break
    reports = [r for r in reports if r is not None]
    schedulereport = {
        "ControlstepReports": reports,
        "ReservoirTime": np.cumsum(step_val),
        "Converged": np.array([r["Converged"] for r in reports]),
        "Iterations": np.array([r["Iterations"] for r in reports]),
        "SimulationTime": simtime,
        "Failure": failure,
    }
    early = ""
    if report["EarlyStop"]:
        n_steps = len(reports)
        early = " (termination triggered by stopFunction) "
    print(f"*** Simulation complete. Solved {n_steps} control steps in "
          f"{format_time_range(float(simtime.sum()))}{early} ***")
    well_sols = [w for w in well_sols if w is not None] if OutputMinisteps else well_sols
    states = [s for s in states if s is not None] if OutputMinisteps else states
    return well_sols, states, schedulereport


def _place(target: List[Any], ind: Sequence[int], values: Sequence[Any]) -> None:
    for k, v in zip(ind, values):
        while k >= len(target):
            target.append(None)
        target[k] = v


def _write_output(handler, offset, restart, positions, values) -> None:
    if handler is None:
        return
    # MRST: `handler{ind + outputOffset - 1}` with a 1-based `ind`; `pos` is
    # 0-based, so the handler id (MRST's file number) is `pos + offset`.
    offset = restart if offset is None else offset
    for pos, v in zip(positions, values):
        handler[pos + offset] = v


def _simulation_header(nstep: int, total_time: float) -> None:
    star = lambda n: "*"*n
    print(f"{star(65)}\n{star(10)} Starting simulation: {nstep:5d} steps, "
          f"{total_time/DAY:5.0f} days {star(9)}\n{star(65)}")


def _step_header(verbose: bool, tm: np.ndarray, restart: int, i: int) -> None:
    n_steps = tm.size - 1
    digits = int(np.floor(np.log10(max(n_steps, 1)))) + 1
    offset = restart - 1
    print(f"Solving timestep {i + 1 + offset:0{digits}d}/{n_steps:0{digits}d}: "
          f"{format_time_range(tm[i + offset], 2)} -> "
          f"{format_time_range(tm[i + 1 + offset], 2)}")


def _disp_step_convergence(its: int, cputime: float) -> None:
    plural = "" if its == 1 else "s"
    print(f"Completed {its} iteration{plural} in {cputime:2.2f} seconds "
          f"({cputime/max(its, 1):2.2f}s per iteration)\n")


simulateScheduleAD = simulate_schedule_ad

__all__ = ["simulate_schedule_ad", "simulateScheduleAD"]


if __name__ == "__main__":
    def test_a_model_that_is_not_a_physical_model_is_refused():
        try:
            simulate_schedule_ad({}, object(), None)
            raise SystemExit("should have raised")
        except AssertionError:
            pass
        print("ok test_a_model_that_is_not_a_physical_model_is_refused")

    test_a_model_that_is_not_a_physical_model_is_refused()
    print("\nAll tests passed!")
