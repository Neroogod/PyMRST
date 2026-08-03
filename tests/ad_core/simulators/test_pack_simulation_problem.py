from types import SimpleNamespace
import importlib
import pickle

import numpy as np
import pytest

from pymrst.modules.ad_core.simulators.sim_runner import (
    PackedProblemManager,
    clear_packed_simulator_output,
    copy_packed_problem,
    get_packed_simulator_output,
    get_multiple_packed_simulator_outputs,
    monitor_background_simulations,
    pack_simulation_problem,
    simulate_packed_problem,
    BackgroundSimulationInfo,
    simulate_packed_problem_background,
    simulate_packed_problem_standalone,
    init_eclipse_packed_problem_ad,
)
from pymrst.modules.ad_core.simulators.sim_runner.packed_problem import ResultHandler
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
    assert isinstance(problem.OutputHandlers.states, ResultHandler)
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


def test_standalone_worker_loads_payload_logs_and_releases_lock(tmp_path):
    schedule = simple_schedule([1.0, 2.0])
    problem = pack_simulation_problem(
        {"x": 0.0}, PickleModel(), schedule, "Standalone",
        directory=tmp_path / "results", name="worker",
    )
    work = tmp_path / "work"
    work.mkdir()
    with (work / "problem.pkl").open("wb") as stream:
        pickle.dump(problem, stream)

    ok, status = simulate_packed_problem_standalone(work)

    assert ok and status.RestartIndex == 0
    assert not (work / "lock.pymrst").exists()
    assert "ok=True" in (work / "simulation.log").read_text()
    assert len(problem.OutputHandlers.states) == 2


def test_init_eclipse_packed_problem_preserves_initialized_setup(
    tmp_path, monkeypatch
):
    module = importlib.import_module(
        "pymrst.modules.ad_core.simulators.sim_runner."
        "init_eclipse_packed_problem_ad"
    )
    state = {"pressure": [1.0]}
    model = SimpleNamespace(inputdata={"RUNSPEC": {"TITLE": "Deck case"}})
    schedule = simple_schedule([1.0])
    nonlinear = SimpleNamespace(tag="solver")
    monkeypatch.setattr(
        module,
        "init_eclipse_problem_ad",
        lambda deck, **options: (state, model, schedule, nonlinear),
    )

    problem = init_eclipse_packed_problem_ad(
        {"deck": True}, name="run", directory=tmp_path
    )

    assert problem.BaseName == "Deck case"
    assert problem.Name == "run"
    assert problem.SimulatorSetup.state0 is state
    assert problem.SimulatorSetup.model is model
    assert problem.SimulatorSetup.schedule is schedule
    assert problem.SimulatorSetup.NonLinearSolver is nonlinear


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


def test_the_observer_hook_is_observational(tmp_path):
    """A handler that tries to interfere must not change the run.

    This runner drives `model.step` itself rather than delegating to
    `NonLinearSolver`, so the canonical instrumentation never reaches it and it
    carried no events at all. The hook added for Milestone 1.4 is checked here
    the same way the solver's is: with a handler that mutates every payload it
    is given and raises on some events, not a passive recorder. A passive
    recorder only shows that a well-behaved handler is harmless, which is the
    easy half of the claim.
    """
    class Model:
        def step(self, state, dt):
            return {"x": state["x"] + dt}, True, 2

    class Hostile:
        def __init__(self):
            self.events, self.mutations, self.types = [], 0, set()

        def __call__(self, kind, payload):
            self.events.append(kind)
            for key, value in list(payload.items()):
                self.types.add(type(value).__name__)
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    payload[key] = float(value) * -1e6
                    self.mutations += 1
            payload["injected"] = "never read"
            if len(self.events) % 2 == 0:
                raise RuntimeError("hostile handler")

    def run(observer, name):
        schedule = SimpleNamespace(step=SimpleNamespace(
            val=[1.0, 2.0, 3.0], control=[1, 1, 1]))
        problem = pack_simulation_problem(
            {"x": 0.0}, Model(), schedule, "Case",
            directory=tmp_path / name, name=name,
        )
        ok, status = simulate_packed_problem(problem, observer=observer)
        _, states, reports, _ = get_packed_simulator_output(problem)
        return ok, status.RestartIndex, [s["x"] for s in states], \
            [r["Converged"] for r in reports]

    plain = run(None, "plain")
    hostile = Hostile()
    observed = run(hostile, "observed")

    assert hostile.events, "the hook never fired, so this proves nothing"
    assert observed == plain, "a hostile handler changed the run"
    # The surface is the real guarantee: a handler cannot corrupt what it is
    # never handed. Anything richer than a scalar here means live state leaked.
    assert hostile.types <= {"int", "float", "bool", "str", "NoneType"}, hostile.types


def test_standalone_worker_records_a_child_event_log_and_manifest(tmp_path):
    """A detached run's only account of itself.

    `simulate_packed_problem_background` forks a child that nothing can watch
    while it runs, and the sole record was one log line naming a restart index
    and a message. The child now also writes `events.jsonl` and
    `run_manifest.json`.

    Both are written with the standard library alone: this module is
    replica-layer and the covenant forbids it importing Research Studio, so the
    event log is JSON Lines by hand and the manifest a plain dict. The
    extension can read either without either side depending on the other.
    """
    import json

    schedule = simple_schedule([1.0, 2.0])
    problem = pack_simulation_problem(
        {"x": 0.0}, PickleModel(), schedule, "Evidence",
        directory=tmp_path / "results", name="child",
    )
    work = tmp_path / "work"
    work.mkdir()
    with (work / "problem.pkl").open("wb") as stream:
        pickle.dump(problem, stream)

    ok, status = simulate_packed_problem_standalone(work)
    assert ok

    info = BackgroundSimulationInfo(path=work)
    records = [json.loads(line)
               for line in info.events_path.read_text(encoding="utf-8").splitlines()]
    assert records, "a completed child wrote no events"
    # Contiguous and ordered: a gap means an event was dropped, and the log is
    # the only place a detached failure can be reconstructed from.
    assert [r["sequence"] for r in records] == list(range(1, len(records)+1))
    assert {"report_step_started", "step_accepted"} <= {r["event"] for r in records}

    manifest = json.loads(info.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1
    assert manifest["base_name"] == "Evidence" and manifest["name"] == "child"
    assert manifest["ok"] is True and manifest["error"] is None
    assert manifest["finished_unix"] >= manifest["started_unix"]
    # The manifest's tally must agree with the log it summarizes.
    counted = {}
    for record in records:
        counted[record["event"]] = counted.get(record["event"], 0) + 1
    assert manifest["event_counts"] == counted


def test_a_crashed_child_still_leaves_a_manifest_naming_the_error(tmp_path):
    """The case the account matters most, and the easy one to omit.

    Writing the manifest only on the success path would leave a crashed child
    indistinguishable from one that never started.
    """
    import json

    work = tmp_path / "work"
    work.mkdir()
    (work / "problem.pkl").write_bytes(b"not a pickle")

    with pytest.raises(Exception):
        simulate_packed_problem_standalone(work)

    info = BackgroundSimulationInfo(path=work)
    assert not info.lock_path.exists(), "a crashed child kept its lock"
    manifest = json.loads(info.manifest_path.read_text(encoding="utf-8"))
    assert manifest["ok"] is False
    assert manifest["error"] and manifest["restart_index"] is None


def test_evidence_recording_can_be_turned_off(tmp_path):
    schedule = simple_schedule([1.0])
    problem = pack_simulation_problem(
        {"x": 0.0}, PickleModel(), schedule, "Quiet", directory=tmp_path / "r",
        name="child",
    )
    work = tmp_path / "work"
    work.mkdir()
    with (work / "problem.pkl").open("wb") as stream:
        pickle.dump(problem, stream)

    ok, _ = simulate_packed_problem_standalone(work, record_evidence=False)
    info = BackgroundSimulationInfo(path=work)
    assert ok
    assert not info.events_path.exists() and not info.manifest_path.exists()
    assert "ok=True" in info.log_path.read_text(encoding="utf-8")
