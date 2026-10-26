"""Batch bookkeeping for packed problems (ad_core port).

A batch runner has one job that is easy to get subtly wrong: knowing how many
simulations are *actually* running. A process can exit between two polls, and
until its slot is reaped it still counts against capacity - so a manager that
counts stale entries reports load that is not there and quietly stops launching
work while nothing runs.

That is why `get_number_of_running_simulations` reaps first and counts second,
and why the ordering is the thing tested here rather than the count.

No real simulation is started. These are stubs, because what is under test is
the bookkeeping, and a batch runner that needs a reservoir simulation to test
its queue logic is a batch runner nobody will test.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.modules.ad_core.simulators.sim_runner.packed_problem_manager_ad import (
    PackedProblemManagerAD,
)


class _Process:
    """A stand-in for a background process, with a settable liveness."""

    def __init__(self, alive=True):
        self._alive = alive
        self.joined = False

    def is_alive(self):
        return self._alive

    def join(self):
        self.joined = True

    def finish(self):
        self._alive = False


def _problem(complete=0, steps=3):
    return SimpleNamespace(
        OutputHandlers=SimpleNamespace(states=[None]*complete),
        SimulatorSetup=SimpleNamespace(
            schedule=SimpleNamespace(step_val=[1.0]*steps)),
    )


def _manager(problems, running=()):
    manager = PackedProblemManagerAD(list(problems),
                                   max_simultaneous_simulations=2)
    for key, process in running:
        manager._background[key] = SimpleNamespace(process=process)
    return manager


# -- the ordering that matters ---------------------------------------------

def test_a_finished_process_is_reaped_before_the_count_is_taken():
    """The whole point.

    Counting first would report one running simulation that has already
    exited, and the batch would refuse to launch anything while nothing runs.
    """
    process = _Process(alive=False)
    manager = _manager([_problem(), _problem()], running=[(0, process)])

    assert manager.get_number_of_running_simulations() == 0
    assert process.joined, "a finished process must be joined, not leaked"
    assert 0 not in manager._background


def test_a_live_process_is_still_counted():
    manager = _manager([_problem(), _problem()],
                       running=[(0, _Process(alive=True))])
    assert manager.get_number_of_running_simulations() == 1


def test_a_problem_that_reached_its_last_step_is_reaped_too():
    """Completion is the other way a slot frees up - the process may still be
    winding down while its work is done."""
    manager = _manager([_problem(complete=3, steps=3)],
                       running=[(0, _Process(alive=True))])
    assert manager.get_number_of_running_simulations() == 0


def test_reaping_leaves_unfinished_work_alone():
    live, dead = _Process(alive=True), _Process(alive=False)
    manager = _manager([_problem(), _problem()],
                       running=[(0, live), (1, dead)])
    manager.update_number_of_running_simulations()
    assert set(manager._background) == {0}


# -- counting completed steps ----------------------------------------------

def test_completed_steps_are_read_from_each_problems_output():
    manager = _manager([_problem(complete=0), _problem(complete=2),
                        _problem(complete=5)])
    np.testing.assert_array_equal(manager.update_number_of_completed_steps(),
                                  [0, 2, 5])


def test_problem_fun_applies_over_the_whole_batch_in_order():
    manager = _manager([_problem(steps=1), _problem(steps=4)])
    np.testing.assert_array_equal(
        manager.problem_fun(lambda p: len(p.SimulatorSetup.schedule.step_val)),
        [1, 4])


# -- launching without the capacity check ----------------------------------

def test_the_escape_hatch_launches_past_the_capacity_limit(monkeypatch):
    """For a caller doing its own scheduling. It must ignore capacity - that
    is the entire reason it exists."""
    import pymrst.modules.ad_core.simulators.sim_runner.packed_problem_manager_ad as mod
    monkeypatch.setattr(mod, "simulate_packed_problem_background_ad",
                        lambda problem, **kw: SimpleNamespace(process=_Process()))

    manager = _manager([_problem(), _problem(), _problem()],
                       running=[(0, _Process()), (1, _Process())])
    assert len(manager._background) == manager.max_simultaneous_simulations

    assert manager.launch_batch_simulation_no_capacity_check(2) == 2
    assert 2 in manager._background


def test_it_still_refuses_to_launch_a_problem_twice(monkeypatch):
    """Two processes against one output handler means two writers on one
    directory, and the results interleave."""
    import pymrst.modules.ad_core.simulators.sim_runner.packed_problem_manager_ad as mod
    monkeypatch.setattr(mod, "simulate_packed_problem_background_ad",
                        lambda problem, **kw: SimpleNamespace(process=_Process()))

    manager = _manager([_problem()], running=[(0, _Process())])
    with pytest.raises(ValueError, match="already running"):
        manager.launch_batch_simulation_no_capacity_check(0)


def test_exactly_one_index_is_required(monkeypatch):
    import pymrst.modules.ad_core.simulators.sim_runner.packed_problem_manager_ad as mod
    monkeypatch.setattr(mod, "simulate_packed_problem_background_ad",
                        lambda problem, **kw: SimpleNamespace(process=_Process()))
    manager = _manager([_problem(), _problem()])
    with pytest.raises(ValueError, match="exactly one"):
        manager.launch_batch_simulation_no_capacity_check([0, 1])


def test_a_boolean_mask_selects_the_problem(monkeypatch):
    import pymrst.modules.ad_core.simulators.sim_runner.packed_problem_manager_ad as mod
    monkeypatch.setattr(mod, "simulate_packed_problem_background_ad",
                        lambda problem, **kw: SimpleNamespace(process=_Process()))
    manager = _manager([_problem(), _problem()])
    assert manager.launch_batch_simulation_no_capacity_check(
        np.array([False, True])) == 1


def test_background_launch_skips_what_is_already_running(monkeypatch):
    import pymrst.modules.ad_core.simulators.sim_runner.packed_problem_manager_ad as mod
    monkeypatch.setattr(mod, "simulate_packed_problem_background_ad",
                        lambda problem, **kw: SimpleNamespace(process=_Process()))

    manager = _manager([_problem(), _problem()], running=[(0, _Process())])
    np.testing.assert_array_equal(manager.simulate_problem_background(), [1])
