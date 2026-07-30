"""Collection and copy helpers for packed simulation problems."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from pymrst.modules.ad_core.simulators.sim_runner.packed_output import (
    get_packed_simulator_output,
)
from pymrst.modules.ad_core.simulators.sim_runner.packed_problem import (
    PackedSimulationProblem,
    ResultHandler,
)


def get_multiple_packed_simulator_outputs(
    problems: Iterable[PackedSimulationProblem], *, min_steps: int = 0,
    read_from_disk: bool = True,
):
    """Collect completed outputs and metadata from several packed problems."""
    all_wells, all_states, all_reports = [], [], []
    names, times, grids, active = [], [], [], []
    for problem in list(problems):
        wells, states, reports, _ = get_packed_simulator_output(
            problem, read_from_disk=read_from_disk
        )
        count = len(problem.OutputHandlers.states)
        is_active = count >= int(min_steps)
        active.append(is_active)
        if not is_active:
            continue
        step = problem.SimulatorSetup.schedule.step
        dt = step["val"] if isinstance(step, dict) else step.val
        all_wells.append(wells)
        all_states.append(states)
        all_reports.append(reports)
        names.append(problem.Name)
        times.append(np.cumsum(np.asarray(dt, dtype=float))[:count])
        model = problem.SimulatorSetup.model
        grids.append(getattr(model, "G", getattr(model, "g", None)))
    return (
        all_wells, all_states, all_reports, names, times, grids,
        np.asarray(active, dtype=bool),
    )


def copy_packed_problem(
    problem: PackedSimulationProblem,
    new_base_name: str,
    *,
    directory: str | Path | None = None,
    name: str | None = None,
    description: str | None = None,
) -> PackedSimulationProblem:
    """Copy problem inputs under a new identity with empty output handlers."""
    new_name = problem.Name if name is None else str(name)
    if directory is None:
        original = problem.OutputHandlers.states.directory
        # original is directory/base-name/problem-name in the standard packer
        directory = original.parent.parent / str(new_base_name)
    root = Path(directory) / new_name
    if (str(new_base_name) == problem.BaseName and new_name == problem.Name
            and root.resolve()
            == problem.OutputHandlers.states.directory.resolve()):
        raise ValueError("copy must not reuse the original output directory")
    prefix = "sub" if problem.SimulatorSetup.OutputMinisteps else ""
    handlers = type(problem.OutputHandlers)(
        states=ResultHandler(root, f"{prefix}state"),
        reports=ResultHandler(root, f"{prefix}report"),
        wellSols=ResultHandler(root, f"{prefix}wellSols"),
    )
    return PackedSimulationProblem(
        BaseName=str(new_base_name),
        Name=new_name,
        Description=(problem.Description if description is None
                     else str(description)),
        SimulatorSetup=problem.SimulatorSetup,
        Modules=list(problem.Modules),
        OutputHandlers=handlers,
    )


getMultiplePackedSimulatorOutputs = get_multiple_packed_simulator_outputs
copyPackedProblem = copy_packed_problem

__all__ = [
    "get_multiple_packed_simulator_outputs",
    "getMultiplePackedSimulatorOutputs",
    "copy_packed_problem",
    "copyPackedProblem",
]
