"""MRST ``ad-core/simulators/sim_runner/PackedProblemManager.m``.

Runs a set of packed problems, in this session (`simulateProblem`) or in
background sessions (`simulateProblemBackground`, or `simulateProblemsBatch`,
which keeps at most `maxSimultaneousSimulations` running, launches the next
when one has written all its states, and reports progress with
`monitorBackgroundSimulations` until all are done). A problem counts as
running from its launch until its state handler holds as many states as its
schedule has steps.

Options are MRST's: `numThreadsPerSimulation` (1), `maxSimultaneousSimulations`
(the number of cores, MRST's `maxNumCompThreads()`), `verbose`,
`displayProgressType` ('graphical' with a MATLAB desktop and 'text' without;
a Python session has none, so 'text'; 'text-scrolling' does not erase), `delay`
between polls (1 s) and `showOnlyActive` (whether progress shows only the
running problems; default: more than 20 problems). Indices are 0-based, and
`None` means all problems (MATLAB's `':'`). MRST starts a single-threaded
background MATLAB with `-singleCompThread`; Python has no such interpreter
option, and the limit is the `maxNumCompThreads` passed to every background
run.

`PackedProblemManagerAD` (in `packed_problem_manager_ad.py`) is PyMRST's
earlier manager, kept for its callers.
"""

from __future__ import annotations

import os
import time
from types import SimpleNamespace
from typing import Any, Callable, Optional, Sequence

import numpy as np

from pymrst.modules.ad_core.ad_oo.physical_model import merge_options
from pymrst.modules.ad_core.simulators.sim_runner.monitor_background_simulations import (
    monitor_background_simulations,
)
from pymrst.modules.ad_core.simulators.sim_runner.simulate_packed_problem import (
    simulate_packed_problem,
)
from pymrst.modules.ad_core.simulators.sim_runner.simulate_packed_problem_background import (
    simulate_packed_problem_background,
)


class PackedProblemManager:
    """MRST `PackedProblemManager` (a handle class: methods update the manager)."""

    def __init__(self, problems: Any, **options: Any):
        self.numThreadsPerSimulation = 1
        self.maxSimultaneousSimulations = os.cpu_count() or 1
        self.verbose = False
        self.displayProgressType = None
        self.delay = 1
        self.showOnlyActive = None
        merge_options(self, options)
        if self.displayProgressType is None:
            self.displayProgressType = "text"
        if not isinstance(problems, (list, tuple)):
            problems = [problems]
        self._packedProblems = list(problems)
        self._N = len(self._packedProblems)
        if self.showOnlyActive is None:
            self.showOnlyActive = self._N > 20
        self._nsteps = self.problemfun(
            lambda x: np.asarray(x.SimulatorSetup.schedule.step_val).size)
        self._problem_is_executing = np.zeros(self._N, dtype=bool)
        self._ncomplete = np.zeros(0, dtype=int)
        self._handle = SimpleNamespace(figure=None, text="", iteration=0)

    def simulate_problems_batch(self, index: Any = None) -> None:
        """Simulate in the background, `maxSimultaneousSimulations` at a time."""
        self._handle = SimpleNamespace(figure=None, text="", iteration=0)
        done = False
        while not done:
            launched, sim_is_done = self.launch_batch_simulations(index)
            done = bool(np.all(sim_is_done))
            self.monitor_progress(True, index)
            if self.delay > 0:
                time.sleep(self.delay)
            if len(launched) and self.verbose:
                print(f"Launched {len(launched)} new sessions...")

    def simulate_problem(self, index: Any = None, **varargin: Any):
        """`simulatePackedProblem` on the problems, in this session."""
        return simulate_packed_problem(self._select(index), **varargin)

    def monitor_progress(self, single_update: bool = False, index: Any = None) -> None:
        problems = self._select(index)
        indices = np.arange(self._N)
        active: Any = slice(None)
        already_done = 0
        if self.showOnlyActive:
            executing = self._problem_is_executing[self._index(index)]
            fh = self._handle.figure
            if executing.any() or fh is not None:
                is_done = self.get_number_of_complete_steps(False) >= self._nsteps
                already_done = int(np.sum(is_done[~executing]))
                active = executing
        indices = indices[active]
        if isinstance(active, slice):
            active_problems = problems
        else:
            active_problems = [p for p, keep in zip(problems, active) if keep]
        kind = self.displayProgressType.lower()
        if kind in ("graphical", "text", "text-scrolling"):
            self._handle = monitor_background_simulations(
                active_problems, useFigure=kind == "graphical", handle=self._handle,
                dynamicText=kind != "text-scrolling", totalNumberOfCases=len(problems),
                totalProgress=already_done, singleUpdate=single_update,
                indices=list(indices + 1))

    def launch_batch_simulations(self, index: Any = None):
        """Launch what capacity allows: `(launched, done)` for `index`."""
        index = self._index(index)
        completed_steps = self.get_number_of_complete_steps(True)
        done = completed_steps[index] >= self._nsteps[index]
        n = self.get_number_of_running_simulations()
        capacity = self.maxSimultaneousSimulations - n
        if capacity > 0:
            candidates = index[~done & ~self._problem_is_executing[index]]
            launched = candidates[:min(capacity, candidates.size)]
            for i in launched:
                self.launch_batch_simulation_no_capacity_check(int(i))
        else:
            launched = np.zeros(0, dtype=int)
        return launched, done

    def launch_batch_simulation_no_capacity_check(self, index: Any) -> None:
        index = np.atleast_1d(self._index(index))
        assert index.size == 1
        index = int(index[0])
        assert not self._problem_is_executing[index]
        nthread = self.numThreadsPerSimulation
        self.simulate_problem_background(index, extra_arg="", maxNumCompThreads=nthread)
        self._problem_is_executing[index] = True

    def get_number_of_running_simulations(self) -> int:
        self.update_number_of_running_simulations()
        return int(np.sum(self._problem_is_executing))

    def update_number_of_running_simulations(self) -> None:
        n = self.get_number_of_complete_steps(True)
        self._problem_is_executing = self._problem_is_executing & ~(n >= self._nsteps)

    def simulate_problem_background(self, index: Any = None, **varargin: Any) -> None:
        for problem in self._select(index):
            simulate_packed_problem_background(problem, **varargin)

    def problemfun(self, fn: Callable[[Any], Any], unif: bool = True) -> Any:
        out = [fn(p) for p in self._packedProblems]
        return np.array(out) if unif else out

    def get_number_of_complete_steps(self, update: bool = True) -> np.ndarray:
        if update:
            self.update_number_of_completed_steps()
        return self._ncomplete

    def update_number_of_completed_steps(self) -> None:
        self._ncomplete = self.problemfun(lambda x: x.OutputHandlers.states.numel_data())

    def get_packed_problems(self, index: Any = None) -> list:
        return self._packedProblems if index is None else self._select(index)

    def _index(self, index: Any) -> np.ndarray:
        if index is None or isinstance(index, str):
            return np.arange(self._N)
        index = np.atleast_1d(np.asarray(index))
        return np.flatnonzero(index) if index.dtype == bool else index.astype(int)

    def _select(self, index: Any) -> list:
        return [self._packedProblems[i] for i in self._index(index)]

    simulateProblemsBatch = simulate_problems_batch
    simulateProblem = simulate_problem
    monitorProgress = monitor_progress
    launchBatchSimulations = launch_batch_simulations
    launchBatchSimulationNoCapacityCheck = launch_batch_simulation_no_capacity_check
    getNumberOfRunningSimulations = get_number_of_running_simulations
    updateNumberOfRunningSimulations = update_number_of_running_simulations
    simulateProblemBackground = simulate_problem_background
    getNumberOfCompleteSteps = get_number_of_complete_steps
    updateNumberOfCompletedSteps = update_number_of_completed_steps
    getPackedProblems = get_packed_problems


__all__ = ["PackedProblemManager"]


if __name__ == "__main__":
    import contextlib
    import io
    from unittest import mock

    class _States:
        def __init__(self, n: int):
            self.n = n

        def numel_data(self) -> int:
            return self.n

    def _problem(name: str, steps: int, stored: int) -> SimpleNamespace:
        return SimpleNamespace(
            BaseName="base", Name=name, Description="",
            SimulatorSetup=SimpleNamespace(schedule=SimpleNamespace(step_val=np.ones(steps))),
            OutputHandlers=SimpleNamespace(states=_States(stored)))

    MODULE = __name__

    def test_options_and_defaults():
        ppm = PackedProblemManager([_problem("a", 3, 0)], maxsimultaneoussimulations=2)
        assert ppm.maxSimultaneousSimulations == 2 and ppm.displayProgressType == "text"
        assert ppm.showOnlyActive is False and list(ppm._nsteps) == [3]
        try:
            PackedProblemManager([], bogus=1)
        except AttributeError:
            pass
        else:
            raise AssertionError("expected AttributeError")
        print("ok test_options_and_defaults")

    def test_batch_launches_within_capacity_until_done():
        problems = [_problem(f"p{i}", 2, 0) for i in range(3)]
        launched = []

        def fake_background(problem, **kw):
            launched.append((problem.Name, kw["maxNumCompThreads"]))
            problem.OutputHandlers.states.n = 2

        ppm = PackedProblemManager(problems, maxSimultaneousSimulations=2, delay=0)
        with mock.patch(f"{MODULE}.simulate_packed_problem_background", fake_background), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            ppm.simulate_problems_batch()
        assert launched == [("p0", 1), ("p1", 1), ("p2", 1)]
        assert "Complete!" in out.getvalue()
        assert ppm.get_number_of_running_simulations() == 0
        print("ok test_batch_launches_within_capacity_until_done")

    def test_capacity_is_respected():
        problems = [_problem(f"p{i}", 2, 0) for i in range(3)]
        ppm = PackedProblemManager(problems, maxSimultaneousSimulations=2, delay=0)
        with mock.patch(f"{MODULE}.simulate_packed_problem_background", lambda p, **k: None):
            launched, done = ppm.launch_batch_simulations()
            assert list(launched) == [0, 1] and not done.any()
            launched, _ = ppm.launch_batch_simulations()
            assert launched.size == 0
        print("ok test_capacity_is_respected")

    def test_simulate_problem_runs_the_selection_in_process():
        problems = [_problem("a", 1, 0), _problem("b", 1, 0)]
        seen = []
        ppm = PackedProblemManager(problems)
        with mock.patch(f"{MODULE}.simulate_packed_problem",
                        lambda p, **kw: seen.append([x.Name for x in p])):
            ppm.simulate_problem([1])
        assert seen == [["b"]]
        print("ok test_simulate_problem_runs_the_selection_in_process")

    test_options_and_defaults()
    test_batch_launches_within_capacity_until_done()
    test_capacity_is_respected()
    test_simulate_problem_runs_the_selection_in_process()
    print("\nAll tests passed!")
