"""Execute and restart disk-backed packed simulation problems."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Iterable

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


def _run_one(problem, restart_step, check_too_many):
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
        candidate, converged, iterations = setup.model.step(
            state, float(dt[index])
        )
        if not converged:
            raise RuntimeError(
                f"packed step {index + 1} failed to converge; "
                f"ministep cutting requires the general schedule driver"
            )
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


def simulate_packed_problem(
    problems: PackedSimulationProblem | Iterable[PackedSimulationProblem],
    *,
    continue_on_error: bool | None = None,
    restart_step: int | None = None,
    check_too_many: bool = True,
):
    """Simulate one or more problems and persist every completed report step."""
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
                problem, restart_step, check_too_many
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


simulatePackedProblem = simulate_packed_problem

__all__ = [
    "PackedSimulationStatus",
    "simulate_packed_problem",
    "simulatePackedProblem",
]
