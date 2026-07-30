from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_core.simulators.sim_runner import (
    PackedProblemManager,
    clear_packed_simulator_output,
    copy_packed_problem,
    get_packed_simulator_output,
    get_multiple_packed_simulator_outputs,
    monitor_background_simulations,
    pack_simulation_problem,
    simulate_packed_problem,
    simulate_packed_problem_background,
)
from pymrst.modules.ad_core.utils.schedule import simple_schedule


class PickleModel:
    def step(self, state, dt):
        return {"x": state["x"] + dt}, True, 1


def test_pack_layout_and_atomic_handlers(tmp_path):
    model = SimpleNamespace(tag="model")
    state = {"pressure": [1.0, 2.0]}
    schedule = SimpleNamespace(step=SimpleNamespace(val=[1.0], control=[1]))
    problem = pack_simulation_problem(
        state, model, schedule, "Case", directory=tmp_path,
        name="solver", description="test case",
    )
    assert problem.BaseName == "Case"
    assert problem.Name == "solver"
    assert problem.Description == "test case"
    assert problem.SimulatorSetup.state0 is state
    assert len(problem.OutputHandlers.states) == 0
    problem.OutputHandlers.states[0] = {"time": 1.0}
    assert problem.OutputHandlers.states[0] == {"time": 1.0}
    assert problem.OutputHandlers.states.indices() == [0]


def test_hash_names_and_ministep_prefixes_are_deterministic(tmp_path):
    args = (
        {"pressure": [1.0]},
        SimpleNamespace(tag="m"),
        SimpleNamespace(step={"val": [1.0]}),
        "Case",
    )
    a = pack_simulation_problem(
        *args, directory=tmp_path, name="run", use_hash=True,
        output_ministeps=True,
    )
    b = pack_simulation_problem(
        *args, directory=tmp_path, name="run", use_hash=True,
        output_ministeps=True,
    )
    assert a.Name == b.Name and a.Name.startswith("run_")
    assert a.OutputHandlers.states.prefix == "substate"
    a.OutputHandlers.states[0] = 1
    a.OutputHandlers.states.reset_data()
    assert len(a.OutputHandlers.states) == 0


def test_output_readback_fallback_and_range_clear(tmp_path):
    problem = pack_simulation_problem(
        {}, SimpleNamespace(), SimpleNamespace(), "Case",
        directory=tmp_path, name="run",
    )
    for index in range(3):
        problem.OutputHandlers.states[index] = {
            "time": index + 1, "wellSol": [f"W{index}"]
        }
        problem.OutputHandlers.reports[index] = {"step": index + 1}
    wells, states, reports, model = get_packed_simulator_output(problem)
    assert wells == [["W0"], ["W1"], ["W2"]]
    assert [state["time"] for state in states] == [1, 2, 3]
    assert [report["step"] for report in reports] == [1, 2, 3]
    assert model is problem.SimulatorSetup.model

    clear_packed_simulator_output(problem, start=1, stop=1)
    assert problem.OutputHandlers.states.indices() == [0, 2]
    assert problem.OutputHandlers.reports.indices() == [0, 2]


def test_simulation_persists_and_auto_restart_is_idempotent(tmp_path):
    class Model:
        def step(self, state, dt):
            return {"x": state["x"] + dt}, True, 2

    schedule = SimpleNamespace(step=SimpleNamespace(
        val=[1.0, 2.0, 3.0], control=[1, 1, 1]
    ))
    problem = pack_simulation_problem(
        {"x": 0.0}, Model(), schedule, "Case",
        directory=tmp_path, name="run",
    )
    ok, status = simulate_packed_problem(problem)
    assert ok and status.RestartIndex == 0
    _, states, reports, _ = get_packed_simulator_output(problem)
    assert [state["x"] for state in states] == [1.0, 3.0, 6.0]
    assert [state["time"] for state in states] == [1.0, 3.0, 6.0]
    assert all(report["Converged"] for report in reports)

    ok, status = simulate_packed_problem(problem)
    assert ok and status.RestartIndex == 3
    assert len(problem.OutputHandlers.states) == 3


def test_explicit_restart_overwrites_tail(tmp_path):
    class Model:
        increment = 1.0

        def step(self, state, dt):
            return {"x": state["x"] + self.increment * dt}, True, 1

    model = Model()
    schedule = SimpleNamespace(step=SimpleNamespace(
        val=[1.0, 1.0, 1.0], control=[1, 1, 1]
    ))
    problem = pack_simulation_problem(
        {"x": 0.0}, model, schedule, "Case",
        directory=tmp_path, name="restart",
    )
    simulate_packed_problem(problem)
    model.increment = 2.0
    simulate_packed_problem(problem, restart_step=1)
    _, states, _, _ = get_packed_simulator_output(problem)
    assert [state["x"] for state in states] == [1.0, 3.0, 5.0]


def test_collection_filters_incomplete_cases_and_copy_is_isolated(tmp_path):
    class Model:
        G = "grid"

        def step(self, state, dt):
            return {"x": state["x"] + dt}, True, 1

    schedule = SimpleNamespace(step=SimpleNamespace(
        val=[1.0, 2.0], control=[1, 1]
    ))
    complete = pack_simulation_problem(
        {"x": 0.0}, Model(), schedule, "Case",
        directory=tmp_path, name="complete",
    )
    empty = pack_simulation_problem(
        {"x": 0.0}, Model(), schedule, "Case",
        directory=tmp_path, name="empty",
    )
    simulate_packed_problem(complete)
    wells, states, reports, names, times, grids, active = (
        get_multiple_packed_simulator_outputs(
            [complete, empty], min_steps=1
        )
    )
    assert names == ["complete"]
    assert active.tolist() == [True, False]
    assert np.array_equal(times[0], [1.0, 3.0])
    assert grids == ["grid"]
    assert len(states[0]) == len(reports[0]) == 2

    copied = copy_packed_problem(
        complete, "Copied", directory=tmp_path / "copied"
    )
    assert copied.BaseName == "Copied" and copied.Name == "complete"
    assert len(copied.OutputHandlers.states) == 0
    assert len(complete.OutputHandlers.states) == 2


def test_background_process_writes_output_and_releases_lock(tmp_path):
    schedule = SimpleNamespace(step=SimpleNamespace(
        val=[1.0, 2.0], control=[1, 1]
    ))
    problem = pack_simulation_problem(
        {"x": 0.0}, PickleModel(), schedule, "Case",
        directory=tmp_path / "results", name="background",
    )
    info = simulate_packed_problem_background(
        problem, workdir=tmp_path / "work"
    )
    status = monitor_background_simulations(
        info, wait=True, timeout=20.0
    )[0]
    assert status["exitcode"] == 0
    assert not status["running"]
    assert "ok=True" in status["log"]
    assert len(problem.OutputHandlers.states) == 2


def test_native_ad_core_schedule_is_accepted(tmp_path):
    problem = pack_simulation_problem(
        {"x": 0.0}, PickleModel(), simple_schedule([1.0, 2.0]),
        "Case", directory=tmp_path, name="native-schedule",
    )
    ok, _ = simulate_packed_problem(problem)
    assert ok
    _, states, _, _ = get_packed_simulator_output(problem)
    assert [state["x"] for state in states] == [1.0, 3.0]


def test_problem_manager_tracks_direct_simulations(tmp_path):
    schedule = simple_schedule([1.0, 1.0])
    problems = [
        pack_simulation_problem(
            {"x": 0.0}, PickleModel(), schedule, "Case",
            directory=tmp_path, name=f"m{i}",
        )
        for i in range(2)
    ]
    manager = PackedProblemManager(problems, max_simultaneous_simulations=1)
    ok, _ = manager.simulate_problem()
    assert ok.tolist() == [True, True]
    assert manager.get_number_of_complete_steps().tolist() == [2, 2]
    assert manager.monitor_progress()["background"] == []
