"""Read and clear disk-backed packed simulator output."""

from __future__ import annotations

from typing import Iterable

from pymrst.modules.ad_core.simulators.sim_runner.packed_problem import (
    PackedSimulationProblem,
)


def get_packed_simulator_output(
    problem: PackedSimulationProblem, *, read_from_disk: bool = True
):
    """Return ``(well_solutions, states, reports, model)`` for a problem."""
    handlers = problem.OutputHandlers
    if not read_from_disk:
        return (handlers.wellSols, handlers.states, handlers.reports,
                problem.SimulatorSetup.model)
    states = [handlers.states[i] for i in handlers.states.indices()]
    reports = [handlers.reports[i] for i in handlers.reports.indices()]
    well_solutions = [
        handlers.wellSols[i] for i in handlers.wellSols.indices()
    ]
    if not well_solutions:
        well_solutions = [
            (state.get("wellSol") if isinstance(state, dict)
             else getattr(state, "wellSol", None))
            for state in states
        ]
    return well_solutions, states, reports, problem.SimulatorSetup.model


def clear_packed_simulator_output(
    problems: PackedSimulationProblem | Iterable[PackedSimulationProblem],
    *,
    start: int = 0,
    stop: int | None = None,
) -> None:
    """Remove stored entries in the inclusive zero-based interval."""
    items = [problems] if isinstance(problems, PackedSimulationProblem) \
        else list(problems)
    if start < 0:
        raise ValueError("start must be non-negative")
    for problem in items:
        for handler in (
            problem.OutputHandlers.states,
            problem.OutputHandlers.reports,
            problem.OutputHandlers.wellSols,
        ):
            selected = [
                index for index in handler.indices()
                if index >= start and (stop is None or index <= stop)
            ]
            handler.reset_data(selected)


getPackedSimulatorOutput = get_packed_simulator_output
clearPackedSimulatorOutput = clear_packed_simulator_output

__all__ = [
    "get_packed_simulator_output",
    "getPackedSimulatorOutput",
    "clear_packed_simulator_output",
    "clearPackedSimulatorOutput",
]
