"""MRST ``ad-core/simulators/sim_runner/simulatePackedProblem.m``.

`simulate_packed_problem` is the port. For each packed problem it finds where
to (re)start from the stored output, then runs `simulateScheduleAD` with the
problem's handlers, so every report step goes to disk as it completes:
- nothing stored: start at step 1;
- all steps stored: nothing to do;
- some stored: continue after the last, or from a requested `restartStep`
  if the step before it is complete.

Errors are fatal for one problem and reported for several
(`continueOnError`). `simulate_packed_problem_ad`, below, is PyMRST's
earlier runner for the legacy models (`model.step` per report step), kept
for its callers.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable, Optional

import numpy as np

from pymrst.modules.ad_core.simulators.sim_runner.packed_problem import (
    PackedSimulationProblem,
)


@dataclass
class PackedSimulationStatus:
    Message: str = ""
    WallTime: float = 0.0
    RestartIndex: float = np.nan


def _timesteps(schedule) -> np.ndarray:
    if hasattr(schedule, "step_val"):
        return np.asarray(schedule.step_val, dtype=float).ravel()
    step = schedule["step"] if isinstance(schedule, dict) else schedule.step
    values = step["val"] if isinstance(step, dict) else step.val
    return np.asarray(values, dtype=float).ravel()


def _set_time(state: Any, time: float):
    if isinstance(state, dict):
        state["time"] = float(time)
    else:
        try:
            setattr(state, "time", float(time))
        except Exception:
            pass


def _well_solution(state: Any):
    return state.get("wellSol") if isinstance(state, dict) \
        else getattr(state, "wellSol", None)


def _run_one(problem, restart_step, check_too_many, observer=None):
    def _observe(event, **payload):
        if observer is None:
            return
        try:
            observer(event, payload)
        except Exception:
            pass

    setup = problem.SimulatorSetup
    dt = _timesteps(setup.schedule)
    states = problem.OutputHandlers.states
    reports = problem.OutputHandlers.reports
    wells = problem.OutputHandlers.wellSols
    present = states.indices()
    if check_too_many and present and max(present) >= dt.size:
        raise RuntimeError(
            f"too much output exists for {problem.Name}; case may be redefined"
        )

    if restart_step is None:
        start = (max(present) + 1) if present else 0
    else:
        start = int(restart_step)
        if start < 0 or start > dt.size:
            raise IndexError("restart_step is outside the schedule")
        for handler in (states, reports, wells):
            handler.reset_data([i for i in handler.indices() if i >= start])
    if start >= dt.size:
        return start, "Complete output found; nothing to do."

    state = setup.state0 if start == 0 else states[start - 1]
    elapsed = float(np.sum(dt[:start]))
    max_cuts = 6
    for arg in setup.ExtraArguments:
        if isinstance(arg, dict) and "max_cuts" in arg:
            max_cuts = int(arg["max_cuts"])
    for index in range(start, dt.size):
        t0 = perf_counter()
        _observe("report_step_started", report_step=int(index),
                 dt=float(dt[index]), restarted_from=int(start))
        candidate, converged, iterations = setup.model.step(
            state, float(dt[index])
        )
        if not converged:
            # Emitted before raising. Only `report_step_started` was reported,
            # so a detached run's event log ended at the step that began and
            # never said whether it finished - the one question the log exists
            # to answer. A step that fails here ends the run, and the observer
            # is the only place that is visible as it happens.
            _observe("step_rejected", report_step=int(index),
                     dt=float(dt[index]), iterations=int(iterations),
                     wall_time=perf_counter()-t0)
            raise RuntimeError(
                f"packed step {index + 1} failed to converge; "
                f"ministep cutting requires the general schedule driver"
            )
        _observe("step_accepted", report_step=int(index), dt=float(dt[index]),
                 iterations=int(iterations), wall_time=perf_counter()-t0)
        state = candidate
        elapsed += float(dt[index])
        _set_time(state, elapsed)
        states[index] = state
        well_sol = _well_solution(state)
        if well_sol is not None:
            wells[index] = well_sol
        reports[index] = {
            "Step": index + 1,
            "Timestep": float(dt[index]),
            "Iterations": int(iterations),
            "Converged": True,
            "WallTime": perf_counter() - t0,
        }
    return start, ""


def simulate_packed_problem_ad(
    problems: PackedSimulationProblem | Iterable[PackedSimulationProblem],
    *,
    continue_on_error: bool | None = None,
    restart_step: int | None = None,
    check_too_many: bool = True,
    observer=None,
):
    """Simulate one or more problems and persist every completed report step.

    ``observer`` is an optional, dependency-free observation hook matching the
    one ``NonLinearSolver`` carries. This runner owns persistence and restart
    around a problem-owned setup and drives ``model.step`` itself, so the
    canonical hook never reaches it and a packed run produced no events at all.

    Event names match the canonical solver's, with ``restarted_from`` added
    because restart position is the thing this runner uniquely decides. It
    receives copied scalars only, its exceptions are swallowed, and the default
    ``None`` leaves behaviour unchanged.
    """
    scalar = isinstance(problems, PackedSimulationProblem)
    items = [problems] if scalar else list(problems)
    if continue_on_error is None:
        continue_on_error = len(items) > 1
    ok, statuses = [], []
    for problem in items:
        began = perf_counter()
        status = PackedSimulationStatus()
        try:
            restart, message = _run_one(
                problem, restart_step, check_too_many, observer
            )
            ok.append(True)
            status.RestartIndex = restart
            status.Message = message
        except Exception as error:
            ok.append(False)
            status.Message = str(error)
            if not continue_on_error:
                raise
        finally:
            status.WallTime = perf_counter() - began
            statuses.append(status)
    if scalar:
        return ok[0], statuses[0]
    return np.asarray(ok, dtype=bool), statuses


def simulate_packed_problem(problems, *, checkTooMany: bool = True,
                            continueOnError: Optional[bool] = None, plot: bool = False,
                            restartStep: float = np.nan):
    """MRST `simulatePackedProblem`: `(ok, status)` per problem."""
    from pymrst.modules.ad_core.ad_oo.simulate_schedule_ad import simulate_schedule_ad
    if not isinstance(problems, (list, tuple)):
        problems = [problems]
    n = len(problems)
    opt = dict(checkTooMany=checkTooMany, restartStep=restartStep,
               continueOnError=n > 1 if continueOnError is None else continueOnError)
    ok = [True]*n
    status = [PackedSimulationStatus() for _ in range(n)]
    for i, problem in enumerate(problems):
        nls = problem.SimulatorSetup.NonLinearSolver
        do_ministeps = problem.SimulatorSetup.OutputMinisteps
        handlers = problem.OutputHandlers
        first = f' Case "{problem.BaseName}" ({problem.Name})'
        second = f' Description: "{problem.Description}"'
        width = max(len(first), len(second))
        line = "*"*(width + 4)
        if i > 0:
            print()
        print(line)
        print(f"*{first:<{width + 1}} *")
        print(f"*{second:<{width + 1}} *")
        print(line)
        state0, model, schedule, restart, offset, msg = _get_restart(problem, opt)
        if np.isnan(restart):
            if opt["continueOnError"]:
                print(msg)
            else:
                raise RuntimeError(msg)
        timer = perf_counter()
        if np.isfinite(restart):
            try:
                extra = dict(problem.SimulatorSetup.ExtraArguments)
                simulate_schedule_ad(state0, model, schedule, NonLinearSolver=nls,
                                     restartStep=int(restart),
                                     OutputHandler=handlers.states,
                                     WellOutputHandler=handlers.wellSols,
                                     ReportHandler=handlers.reports,
                                     outputOffset=offset, OutputMinisteps=do_ministeps,
                                     **extra)
            except Exception as ex:
                msg = str(ex)
                ok[i] = False
                print(f"!!! Simulation resulted in fatal error !!!\n Exception thrown: {msg}")
                if not opt["continueOnError"]:
                    raise
        status[i].Message = msg
        status[i].WallTime = perf_counter() - timer
        status[i].RestartIndex = restart
    if plot:
        from pymrst.modules.ad_core.simulators.sim_runner.plot_packed_problem import (
            plot_packed_problem,
        )
        plot_packed_problem(problem)
    return ok, status


def _with_time(state: Any, t: float) -> Any:
    """A copy of `state` with `time` set; dict or struct-like states."""
    from copy import copy
    state = copy(state)
    if isinstance(state, dict):
        state["time"] = t
    else:
        setattr(state, "time", t)
    return state


def _time_of(state: Any) -> float:
    return state["time"] if isinstance(state, dict) else getattr(state, "time")


def _get_restart(problem, opt):
    """MRST's `getRestart`: `(state0, model, schedule, restart, offset, msg)`
    with `restart` 1-based, `inf` when complete and `nan` on failure."""
    from copy import copy
    setup = problem.SimulatorSetup
    schedule, model = setup.schedule, setup.model
    state0 = copy(setup.state0)
    T = np.cumsum(_timesteps(schedule))
    handler = problem.OutputHandlers.states
    do_ministeps = setup.OutputMinisteps
    nstep = T.size
    ndata = handler.numel_data()
    msg = ""
    offset = None
    if opt["checkTooMany"] and not do_ministeps:
        assert ndata <= nstep, (f"Too much data exists for {problem.Name}! "
                                "Problem may have been redefined.")
    restart = opt["restartStep"]
    if ndata == 0 or restart <= 1:
        state0 = _with_time(state0, 0)        # MATLAB: `state0.time = 0` on its copy
        endstate = state0
    else:
        endstate = handler[ndata]
    auto_restart = np.isnan(restart)
    if auto_restart:
        if ndata == 0:
            restart = 1
        elif (ndata >= nstep and not do_ministeps) or \
                (_time_of(endstate) >= T[-1] and do_ministeps):
            restart = np.inf
        elif do_ministeps:
            last = np.flatnonzero(T <= _time_of(endstate))
            if last.size == 0:
                restart = 1
            else:
                restart = int(last[-1]) + 2
                state0, offset = handler.get_by_time(T[last[-1]])
                offset = offset + 1
        else:
            restart = ndata + 1
            state0 = endstate
    elif restart > 1:
        t_prev = T[int(restart) - 2]
        if _time_of(endstate) < t_prev:
            msg = (f"Restart step {int(restart)} was specified, but step "
                   f"{int(restart) - 1} is not complete! Aborting.")
            restart = np.nan
        elif do_ministeps:
            state0, offset = handler.get_by_time(t_prev)
            offset = offset + 1
        else:
            state0 = handler[int(restart) - 1]
    if state0 is None:
        restart = np.nan
        msg = "Internal error. Restart state not found."
    if np.isinf(restart):
        print("-> Complete output found, nothing to do here.")
    elif restart == 1:
        print("-> No output found. Starting from first step..." if auto_restart
              else "-> Starting from first step as requested...")
    elif not np.isnan(restart):
        how = "" if auto_restart else "requested "
        print(f"-> Partial output found. Starting from {how}step {int(restart)} of {nstep}...")
    return state0, model, schedule, restart, offset, msg


simulatePackedProblem = simulate_packed_problem

__all__ = [
    "PackedSimulationStatus",
    "simulate_packed_problem",
    "simulate_packed_problem_ad",
    "simulatePackedProblem",
]


if __name__ == "__main__":
    from types import SimpleNamespace

    def test_the_start_time_is_set_on_a_copy():
        """MATLAB's `state0.time = 0` changes its own copy, for dict and
        struct-like states alike."""
        d = {"pressure": 1.0}
        out = _with_time(d, 0)
        assert out["time"] == 0 and "time" not in d
        s = SimpleNamespace(pressure=1.0)
        out = _with_time(s, 0)
        assert _time_of(out) == 0 and not hasattr(s, "time")
        print("ok test_the_start_time_is_set_on_a_copy")

    test_the_start_time_is_set_on_a_copy()
    print("\nAll tests passed!")
