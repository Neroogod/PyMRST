"""Manage direct and capacity-limited background packed simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from time import sleep

import numpy as np

from pymrst.modules.ad_core.simulators.sim_runner.background import (
    monitor_background_simulations,
    simulate_packed_problem_background,
)
from pymrst.modules.ad_core.simulators.sim_runner.simulate_packed_problem import (
    simulate_packed_problem,
)


@dataclass
class PackedProblemManager:
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

    def get_packed_problems(self, index=None):
        if index is None:
            return list(self.problems)
        return [self.problems[i] for i in np.atleast_1d(index)]

    getPackedProblems = get_packed_problems

    def simulate_problem(self, index=None, **options):
        selected = self.problems if index is None else self.get_packed_problems(index)
        return simulate_packed_problem(selected, **options)

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
            info = simulate_packed_problem_background(
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
            "background": monitor_background_simulations(
                list(self._background.values())
            ) if self._background else [],
        }

    monitorProgress = monitor_progress


__all__ = ["PackedProblemManager"]
