"""PyMRST's earlier packed-problem manager, kept for its callers.

The port of MRST `PackedProblemManagerAD` is `packed_problem_manager.py`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import sleep

import numpy as np

from pymrst.modules.ad_core.simulators.sim_runner.background import (
    monitor_background_simulations_ad,
    simulate_packed_problem_background_ad,
)
from pymrst.modules.ad_core.simulators.sim_runner.simulate_packed_problem import (
    simulate_packed_problem_ad,
)


@dataclass
class PackedProblemManagerAD:
    problems: list
    max_simultaneous_simulations: int = 1
    delay: float = 0.1
    workdir: str | Path | None = None
    _background: dict[int, object] = field(default_factory=dict, init=False)

    def __post_init__(self):
        if not isinstance(self.problems, list):
            self.problems = [self.problems]
        if self.max_simultaneous_simulations < 1:
            raise ValueError("max_simultaneous_simulations must be positive")

    def get_number_of_complete_steps(self) -> np.ndarray:
        return np.array(
            [len(problem.OutputHandlers.states) for problem in self.problems],
            dtype=int,
        )

    getNumberOfCompleteSteps = get_number_of_complete_steps

    # -- MRST PackedProblemManagerAD.m, ported ------------------------------

    def problem_fun(self, function):
        """Apply `function` to every packed problem, in order.

        MRST's `problemfun`. It exists so the bookkeeping below reads as one
        line each instead of repeating the loop, and so a caller can ask an
        arbitrary question of the whole batch.
        """
        return np.array([function(problem) for problem in self.problems])

    problemfun = problem_fun

    def update_number_of_completed_steps(self) -> np.ndarray:
        """Refresh and cache how many steps each problem has written."""
        self.ncomplete = self.problem_fun(
            lambda p: len(p.OutputHandlers.states))
        return self.ncomplete

    updateNumberOfCompletedSteps = update_number_of_completed_steps

    def _target_steps(self) -> np.ndarray:
        return self.problem_fun(_schedule_length)

    def update_number_of_running_simulations(self) -> None:
        """Reap simulations that have finished but not yet been cleared.

        A process may exit between two polls, and until it is reaped its slot
        still counts against capacity. Leaving stale entries in place
        over-reports the load and starves the queue - the batch stops launching
        work while nothing is running.
        """
        complete = self.get_number_of_complete_steps()
        targets = self._target_steps()
        for key, info in list(self._background.items()):
            finished = (info.process is not None
                        and not info.process.is_alive())
            if finished or complete[key] >= targets[key]:
                if info.process is not None:
                    info.process.join()
                del self._background[key]

    updateNumberOfRunningSimulations = update_number_of_running_simulations

    def get_number_of_running_simulations(self) -> int:
        """How many simulations are actually running.

        Reaps first, then counts. Counting stale flags would report load that
        is not there, which is the difference between a batch that keeps its
        slots full and one that quietly stalls.
        """
        self.update_number_of_running_simulations()
        return len(self._background)

    getNumberOfRunningSimulations = get_number_of_running_simulations

    def simulate_problem_background(self, index=None, **options):
        """Launch one or more problems in the background, without waiting."""
        selected = (range(len(self.problems)) if index is None
                    else np.atleast_1d(index).astype(int))
        launched = []
        for key in selected:
            key = int(key)
            if key in self._background:
                continue
            self._background[key] = simulate_packed_problem_background_ad(
                self.problems[key], workdir=self.workdir, **options)
            launched.append(key)
        return np.asarray(launched, dtype=int)

    simulateProblemBackground = simulate_problem_background

    def launch_batch_simulation_no_capacity_check(self, index, **options) -> int:
        """Launch exactly one problem, ignoring the capacity limit.

        The escape hatch MRST keeps for a caller doing its own scheduling. It
        still refuses to launch a problem that is already running - starting a
        second process against the same output handler would have two writers
        on one directory, and the results would interleave.
        """
        index = np.atleast_1d(index)
        if index.dtype == bool:
            index = np.flatnonzero(index)
        if index.size != 1:
            raise ValueError(
                f"expected exactly one problem index, got {index.size}")
        key = int(index[0])
        if key in self._background:
            raise ValueError(
                f"problem {key} is already running; launching it twice would "
                f"put two writers on one output directory")
        self._background[key] = simulate_packed_problem_background_ad(
            self.problems[key], workdir=self.workdir, **options)
        return key

    launchBatchSimulationNoCapacityCheck = launch_batch_simulation_no_capacity_check

    def get_packed_problems(self, index=None):
        if index is None:
            return list(self.problems)
        return [self.problems[i] for i in np.atleast_1d(index)]

    getPackedProblems = get_packed_problems

    def simulate_problem(self, index=None, **options):
        selected = self.problems if index is None else self.get_packed_problems(index)
        return simulate_packed_problem_ad(selected, **options)

    simulateProblem = simulate_problem

    def launch_batch_simulations(self, index=None):
        selected = (range(len(self.problems)) if index is None
                    else np.atleast_1d(index).astype(int))
        # Drop completed processes before calculating capacity.
        for key, info in list(self._background.items()):
            if info.process is not None and not info.process.is_alive():
                info.process.join()
                del self._background[key]
        launched = []
        capacity = self.max_simultaneous_simulations - len(self._background)
        complete = self.get_number_of_complete_steps()
        for key in selected:
            key = int(key)
            schedule = self.problems[key].SimulatorSetup.schedule
            nstep = (len(schedule.step_val) if hasattr(schedule, "step_val")
                     else len(schedule.step.val))
            if capacity <= 0 or key in self._background or complete[key] >= nstep:
                continue
            info = simulate_packed_problem_background_ad(
                self.problems[key], workdir=self.workdir
            )
            self._background[key] = info
            launched.append(key)
            capacity -= 1
        done = np.array([
            complete[i] >= (
                len(p.SimulatorSetup.schedule.step_val)
                if hasattr(p.SimulatorSetup.schedule, "step_val")
                else len(p.SimulatorSetup.schedule.step.val)
            )
            for i, p in enumerate(self.problems)
        ])
        return np.asarray(launched, dtype=int), done

    launchBatchSimulations = launch_batch_simulations

    def simulate_problems_batch(self):
        while True:
            self.launch_batch_simulations()
            complete = self.get_number_of_complete_steps()
            targets = np.array([
                len(p.SimulatorSetup.schedule.step_val)
                if hasattr(p.SimulatorSetup.schedule, "step_val")
                else len(p.SimulatorSetup.schedule.step.val)
                for p in self.problems
            ])
            if np.all(complete >= targets) and not self._background:
                return
            sleep(max(float(self.delay), 0.01))

    simulateProblemsBatch = simulate_problems_batch

    def monitor_progress(self):
        return {
            "complete": self.get_number_of_complete_steps(),
            "background": monitor_background_simulations_ad(
                list(self._background.values())
            ) if self._background else [],
        }

    monitorProgress = monitor_progress


__all__ = ["PackedProblemManagerAD"]


def _schedule_length(problem) -> int:
    """How many steps a packed problem's schedule declares."""
    schedule = problem.SimulatorSetup.schedule
    if hasattr(schedule, "step_val"):
        return len(schedule.step_val)
    return len(schedule.step.val)
