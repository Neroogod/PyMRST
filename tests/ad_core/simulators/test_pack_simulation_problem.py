from types import SimpleNamespace

from pymrst.modules.ad_core.simulators.sim_runner import (
    clear_packed_simulator_output,
    get_packed_simulator_output,
    pack_simulation_problem,
    simulate_packed_problem,
)


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
