"""MRST ``ad-core/simulators/sim_runner/getMultiplePackedSimulatorOutputs.m``.

`getPackedSimulatorOutput` for several packed problems, with each problem's
name, output times and grid. Times are report times, or the ministep times
when the problem stored ministeps. Problems with fewer than `minSteps`
outputs are left out; `act` says which were kept.
`get_multiple_packed_simulator_outputs_ad` (in `packed_collection.py`) is
PyMRST's earlier version, kept for its callers.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pymrst.modules.ad_core.simulators.sim_runner.get_packed_simulator_output import (
    get_packed_simulator_output,
)


def get_multiple_packed_simulator_outputs(problems: Sequence[Any], *, minSteps: int = 0,
                                          **arg: Any):
    """MRST `getMultiplePackedSimulatorOutputs`:
    `(all_ws, all_states, all_reports, names, T, grids, act)`."""
    n = len(problems)
    counts = np.zeros(n, dtype=np.int64)
    all_ws, all_states, all_reports, names, T, grids = ([None]*n for _ in range(6))
    for i, problem in enumerate(problems):
        setup = problem.SimulatorSetup
        ministeps = setup.OutputMinisteps
        all_ws[i], all_states[i], all_reports[i], _ = get_packed_simulator_output(problem, **arg)
        names[i] = problem.Name
        counts[i] = _numel_data(all_ws[i])
        schedule_time = np.cumsum(setup.schedule.step_val)
        if ministeps:
            from pymrst.modules.ad_core.utils.convert_report_to_schedule import (
                convert_report_to_schedule,
            )
            _, dt = convert_report_to_schedule(all_reports[i], setup.schedule)
            time = np.cumsum(dt)
        else:
            time = schedule_time
        T[i] = time[:counts[i]]
        grids[i] = setup.model.G
        if ministeps:
            counts[i] = int(np.flatnonzero(time <= T[i][-1])[-1]) + 1
    act = counts >= minSteps
    keep = lambda items: [x for x, a in zip(items, act) if a]
    return (keep(all_ws), keep(all_states), keep(all_reports), keep(names), keep(T),
            keep(grids), act)


def _numel_data(item: Any) -> int:
    return item.numel_data() if hasattr(item, "numel_data") else len(item)


getMultiplePackedSimulatorOutputs = get_multiple_packed_simulator_outputs

__all__ = ["getMultiplePackedSimulatorOutputs", "get_multiple_packed_simulator_outputs"]
