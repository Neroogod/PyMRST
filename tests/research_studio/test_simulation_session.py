"""Observation-only integration tests for the canonical session adapter."""

from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.research_studio import (
    CancellationRequested, CancellationToken, EventBus, EventKind,
    HDF5ResultStore, SimulationSession, SimulationStatus,
)
from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.modules.ad_core.utils.schedule import Control, Schedule
from pymrst.params.wells_and_bc import BoundaryCondition
from pymrst.utils.adi import init_variables_adi, value


@dataclass
class _Step:
    converged: bool = True
    iterations: int = 2
    dt: float = 10.0
    residuals: list = field(default_factory=lambda: [{"mass": 1.0}, {"mass": 0.01}])
    cuts: int = 1
    line_searches: int = 0
    ministeps: list = field(default_factory=lambda: [5.0, 5.0])


@dataclass
class _Report:
    steps: list

    @property
    def converged(self):
        return all(step.converged for step in self.steps)


class _Solver:
    def __init__(self, states, report):
        self.states, self.report, self.calls = states, report, []

    def simulate_schedule(self, model, initial_state, schedule):
        self.calls.append((model, initial_state, schedule))
        return self.states, self.report


def test_session_delegates_once_and_preserves_native_outputs():
    model, state0, schedule, final = object(), object(), [10.0], object()
    report = _Report([_Step()])
    solver = _Solver([final], report)
    result = SimulationSession(
        model=model, initial_state=state0, schedule=schedule,
        nonlinear_solver=solver, case_id="case_test", run_id="run_test",
    ).run()

    assert solver.calls == [(model, state0, schedule)]
    assert result.states is solver.states
    assert result.report is report
    assert result.final_state is final
    assert result.status is SimulationStatus.COMPLETED
    assert [event.sequence for event in result.events] == list(range(len(result.events)))
    assert result.events[0].kind is EventKind.RUN_STARTED
    assert result.events[-1].kind is EventKind.RUN_COMPLETED
    assert result.evidence.timestep_reports[0]["residuals"] == _Step().residuals


def test_handler_failure_cannot_change_solver_call_or_result():
    solver = _Solver(["accepted"], _Report([_Step()]))
    bus = EventBus()
    bus.subscribe(lambda _event: (_ for _ in ()).throw(RuntimeError("UI failed")))
    result = SimulationSession(
        model="model", initial_state="initial", schedule=[10.0],
        nonlinear_solver=solver, event_bus=bus,
    ).run()
    assert result.final_state == "accepted"
    assert len(solver.calls) == 1
    assert bus.failures


def test_pre_run_cancellation_never_calls_solver():
    token = CancellationToken()
    token.request()
    solver = _Solver([], _Report([]))
    session = SimulationSession(
        model="model", initial_state="initial", schedule=[],
        nonlinear_solver=solver, cancellation=token,
    )
    with pytest.raises(CancellationRequested):
        session.run()
    assert solver.calls == []


def test_pre_run_cancellation_does_not_create_store(tmp_path):
    token = CancellationToken()
    token.request()
    store = HDF5ResultStore(tmp_path/"cancelled.h5")
    session = SimulationSession(
        model="model", initial_state={}, schedule=[],
        nonlinear_solver=_Solver([], _Report([])), cancellation=token,
        result_store=store,
    )
    with pytest.raises(CancellationRequested):
        session.run()
    assert not store.path.exists()


def test_nonconverged_native_report_is_returned_as_failed_result():
    report = _Report([_Step(converged=False)])
    solver = _Solver(["last_accepted"], report)
    result = SimulationSession(
        model="model", initial_state="initial", schedule=[10.0],
        nonlinear_solver=solver,
    ).run()
    assert result.report is report
    assert result.status is SimulationStatus.FAILED
    assert result.events[-1].kind is EventKind.RUN_FAILED
    assert result.evidence.failures[-1]["type"] == "NumericalFailureAnalysis"
    assert result.evidence.failures[-1]["analysis"]["classifications"]


def test_real_nonlinear_solver_observation_matches_unobserved_run_exactly():
    class ScalarModel:
        @staticmethod
        def get_equations(state0, state, dt):
            (x,) = init_variables_adi(state["x"])
            return [x*x - 4.0]

        @staticmethod
        def get_convergence(equations, dt):
            residual = float(np.max(np.abs(value(equations[0]))))
            return residual < 1e-10, {"residual": residual}

        @staticmethod
        def update_state(state, increment):
            updated = state.copy()
            updated["x"] = state["x"] + increment
            return updated

    model = ScalarModel()
    initial = State(x=np.array([1.0]))
    schedule = [1.0, 1.0, 1.0]
    native_states, native_report = NonLinearSolver().simulate_schedule(
        model, initial, schedule,
    )
    observed = SimulationSession(
        model=model,
        initial_state=initial,
        schedule=schedule,
        nonlinear_solver=NonLinearSolver(),
        run_id="run_characterization",
    ).run()

    np.testing.assert_array_equal(
        [state["x"][0] for state in observed.states],
        [state["x"][0] for state in native_states],
    )
    assert [step.iterations for step in observed.report.steps] == [
        step.iterations for step in native_report.steps
    ]
    assert [step.cuts for step in observed.report.steps] == [
        step.cuts for step in native_report.steps
    ]
    assert observed.evidence.to_json()


def test_session_transactionally_persists_only_converged_states(tmp_path):
    store = HDF5ResultStore(tmp_path/"session.h5")
    report = _Report([_Step(converged=True), _Step(converged=False)])
    solver = _Solver([
        {"pressure": np.array([10.0])},
        {"pressure": np.array([-99.0])},
    ], report)
    result = SimulationSession(
        model="model", initial_state={"pressure": np.array([11.0])},
        schedule=[10.0, 10.0], nonlinear_solver=solver,
        run_id="run_persisted", result_store=store,
    ).run()

    assert result.status is SimulationStatus.FAILED
    assert store.state_indices("run_persisted") == (0,)
    np.testing.assert_array_equal(
        store.load_state("run_persisted", 0)["pressure"], [10.0],
    )
    info = store.inspect_run("run_persisted")
    assert info["rejected_attempts"] == 1
    assert info["status"] == "failed"
    assert info["complete"]
    records = store.load_event_records("run_persisted")
    assert records[0]["kind"] == "run_started"
    assert records[-1]["kind"] == "run_failed"


def test_solver_exception_leaves_a_finalized_failure_record(tmp_path):
    class ExplodingSolver:
        @staticmethod
        def simulate_schedule(model, initial_state, schedule):
            raise RuntimeError("linear solver exploded")

    store = HDF5ResultStore(tmp_path/"failure.h5")
    session = SimulationSession(
        model="model", initial_state={}, schedule=[1.0],
        nonlinear_solver=ExplodingSolver(), run_id="run_failure",
        result_store=store,
    )
    with pytest.raises(RuntimeError, match="exploded"):
        session.run()
    info = store.inspect_run("run_failure")
    assert info["status"] == "failed" and info["complete"]
    evidence = store.load_evidence("run_failure")
    assert evidence.failures[0]["type"] == "RuntimeError"


def test_failed_session_writes_independently_loadable_bundle(tmp_path):
    from pymrst.research_studio import load_failure_bundle

    report = _Report([_Step(converged=False)])
    solver = _Solver([{"pressure": np.array([9.0])}], report)
    store = HDF5ResultStore(tmp_path/"bundle-store.h5")
    bundle_path = tmp_path/"failure.pymrst-failure"
    result = SimulationSession(
        model="model", initial_state={"pressure": np.array([10.0])},
        schedule=[1.0], checkpoint_schedule=[1.0],
        nonlinear_solver=solver, run_id="run_bundled_failure",
        result_store=store, checkpoint_every_report=True,
        code_fingerprint="code_test", failure_bundle_path=bundle_path,
    ).run()
    assert result.status is SimulationStatus.FAILED
    loaded = load_failure_bundle(bundle_path)
    # The failed candidate is not accepted; replay starts from checkpoint zero.
    assert loaded["checkpoint"].completed_report_steps == 0
    np.testing.assert_array_equal(loaded["checkpoint"].state["pressure"], [10.0])
    assert loaded["failing_step"]["converged"] is False
    assert any(event.kind is EventKind.FAILURE_BUNDLE_WRITTEN for event in result.events)


def test_session_checkpoints_and_branches_with_compatible_lineage(tmp_path):
    class TimeModel:
        @staticmethod
        def get_equations(state0, state, dt):
            (x,) = init_variables_adi(state["x"])
            return [x - state0["x"] - dt]

        @staticmethod
        def get_convergence(equations, dt):
            residual = float(np.max(np.abs(value(equations[0]))))
            return residual < 1e-12, {"residual": residual}

        @staticmethod
        def update_state(state, increment):
            updated = state.copy()
            updated["x"] = state["x"] + increment
            return updated

    model, schedule = TimeModel(), np.array([1.0, 2.0, 3.0, 4.0])
    store = HDF5ResultStore(tmp_path/"branch.h5")
    source = SimulationSession(
        model=model, initial_state=State(x=np.array([0.0])),
        schedule=schedule[:2], nonlinear_solver=NonLinearSolver(),
        run_id="run_source", result_store=store,
        checkpoint_every_report=True, code_fingerprint="code_test",
        checkpoint_schedule=schedule,
    )
    prefix = source.run()
    assert prefix.final_state["x"][0] == 3.0
    assert store.inspect_run("run_source")["checkpoints"] == 2

    branch = SimulationSession.from_checkpoint(
        source_store=store,
        source_run_id="run_source",
        checkpoint_id="checkpoint_00000002",
        model=model,
        full_schedule=schedule,
        nonlinear_solver=NonLinearSolver(),
        code_fingerprint="code_test",
        state_factory=State,
        run_id="run_branch",
        result_store=store,
        checkpoint_every_report=True,
    ).run()
    assert branch.final_state["x"][0] == 10.0
    assert branch.evidence.manifest["lineage"] == {
        "source_run_id": "run_source",
        "source_checkpoint_id": "checkpoint_00000002",
    }
    assert branch.evidence.checkpoints[-1]["completed_report_steps"] == 4


def test_structured_schedule_restart_restores_control_and_matches_full_run(tmp_path):
    class ControlledModel:
        def __init__(self):
            self.bc = BoundaryCondition(value=np.array([1.0]))
            self.facility = SimpleNamespace(wells=[])

        def get_equations(self, state0, state, dt):
            (x,) = init_variables_adi(state["x"])
            return [x - state0["x"] - dt*float(self.bc.value[0])]

        @staticmethod
        def get_convergence(equations, dt):
            residual = float(np.max(np.abs(value(equations[0]))))
            return residual < 1e-12, {"residual": residual}

        @staticmethod
        def update_state(state, increment):
            updated = state.copy()
            updated["x"] = state["x"] + increment
            return updated

    controls = [
        Control(bc=BoundaryCondition(value=np.array([1.0]))),
        Control(bc=BoundaryCondition(value=np.array([2.0]))),
    ]
    full_schedule = Schedule(
        step_val=np.ones(4),
        step_control=np.array([0, 0, 1, 1]),
        controls=controls,
    )
    uninterrupted = SimulationSession(
        model=ControlledModel(), initial_state=State(x=np.array([0.0])),
        schedule=full_schedule, nonlinear_solver=NonLinearSolver(),
        run_id="run_structured_full",
    ).run()

    prefix_schedule = Schedule(
        step_val=full_schedule.step_val[:2],
        step_control=full_schedule.step_control[:2],
        controls=controls,
    )
    store = HDF5ResultStore(tmp_path/"structured.h5")
    SimulationSession(
        model=ControlledModel(), initial_state=State(x=np.array([0.0])),
        schedule=prefix_schedule, checkpoint_schedule=full_schedule,
        nonlinear_solver=NonLinearSolver(), run_id="run_structured_source",
        result_store=store, checkpoint_every_report=True,
        code_fingerprint="code_test",
    ).run()
    resumed = SimulationSession.from_checkpoint(
        source_store=store, source_run_id="run_structured_source",
        checkpoint_id="checkpoint_00000002", model=ControlledModel(),
        full_schedule=full_schedule, nonlinear_solver=NonLinearSolver(),
        code_fingerprint="code_test", state_factory=State,
        run_id="run_structured_branch", result_store=store,
        checkpoint_every_report=True,
    ).run()
    np.testing.assert_array_equal(
        resumed.final_state["x"], uninterrupted.final_state["x"],
    )
    assert resumed.final_state["x"][0] == 6.0


def test_live_solver_observer_is_numerically_inert_and_failure_isolated():
    class ScalarModel:
        @staticmethod
        def get_equations(state0, state, dt):
            (x,) = init_variables_adi(state["x"])
            return [x*x - 4.0]

        @staticmethod
        def get_convergence(equations, dt):
            residual = float(np.max(np.abs(value(equations[0]))))
            return residual < 1e-10, {"residual": residual}

        @staticmethod
        def update_state(state, increment):
            updated = state.copy()
            updated["x"] = state["x"] + increment
            return updated

    model, initial, schedule = ScalarModel(), State(x=np.array([1.0])), [1.0, 1.0]
    native_states, native_report = NonLinearSolver().simulate_schedule(
        model, initial, schedule,
    )
    observed_events = []
    observed_states, observed_report = NonLinearSolver(
        observer=lambda kind, payload: observed_events.append((kind, payload)),
    ).simulate_schedule(model, initial, schedule)
    broken = NonLinearSolver(
        observer=lambda *_args: (_ for _ in ()).throw(RuntimeError("observer failed")),
    )
    broken_states, broken_report = broken.simulate_schedule(model, initial, schedule)

    for candidate in (observed_states, broken_states):
        np.testing.assert_array_equal(
            [state["x"][0] for state in candidate],
            [state["x"][0] for state in native_states],
        )
    assert [step.iterations for step in observed_report.steps] == [
        step.iterations for step in native_report.steps
    ]
    assert [step.iterations for step in broken_report.steps] == [
        step.iterations for step in native_report.steps
    ]
    assert any(kind == "nonlinear_iteration_completed" for kind, _ in observed_events)
    assert any(kind == "linear_solve_completed" for kind, _ in observed_events)
    assert broken.observer_failures


def test_session_live_observation_composes_and_restores_existing_observer():
    class ScalarModel:
        @staticmethod
        def get_equations(state0, state, dt):
            (x,) = init_variables_adi(state["x"])
            return [x*x - 4.0]

        @staticmethod
        def get_convergence(equations, dt):
            residual = float(np.max(np.abs(value(equations[0]))))
            return residual < 1e-10, {"residual": residual}

        @staticmethod
        def update_state(state, increment):
            updated = state.copy()
            updated["x"] = state["x"] + increment
            return updated

    existing_events = []
    existing = lambda kind, payload: existing_events.append(kind)
    solver = NonLinearSolver(observer=existing)
    result = SimulationSession(
        model=ScalarModel(), initial_state=State(x=np.array([1.0])),
        schedule=[1.0], nonlinear_solver=solver,
        run_id="run_live", live_observation=True,
    ).run()
    assert solver.observer is existing
    live = [event for event in result.events if event.payload.get("live")]
    assert any(event.kind is EventKind.NONLINEAR_ITERATION_COMPLETED for event in live)
    assert any(event.kind is EventKind.LINEAR_SOLVE_COMPLETED for event in live)
    correlated = [
        event for event in live
        if event.kind in {
            EventKind.MINISTEP_STARTED,
            EventKind.NONLINEAR_ITERATION_COMPLETED,
            EventKind.LINEAR_SOLVE_COMPLETED,
        }
    ]
    assert correlated
    assert all(event.report_step_id is not None for event in correlated)
    assert all(event.ministep_id is not None for event in correlated)
    assert all(event.attempt_id is not None for event in correlated)
    report_starts = [
        event for event in result.events
        if event.kind is EventKind.REPORT_STEP_STARTED
    ]
    assert len(report_starts) == 1 and report_starts[0].payload["live"]
    assert existing_events
    assert result.evidence.convergence["live_iterations"]
    assert result.evidence.to_json()


def test_streaming_cancellation_commits_boundary_and_resumes_exactly(tmp_path):
    class TimeModel:
        @staticmethod
        def get_equations(state0, state, dt):
            (x,) = init_variables_adi(state["x"])
            return [x - state0["x"] - dt]

        @staticmethod
        def get_convergence(equations, dt):
            residual = float(np.max(np.abs(value(equations[0]))))
            return residual < 1e-12, {"residual": residual}

        @staticmethod
        def update_state(state, increment):
            updated = state.copy()
            updated["x"] = state["x"] + increment
            return updated

    schedule = np.array([1.0, 2.0, 3.0, 4.0])
    model = TimeModel()
    uninterrupted = SimulationSession(
        model=model, initial_state=State(x=np.array([0.0])),
        schedule=schedule, nonlinear_solver=NonLinearSolver(),
    ).run()

    token, bus = CancellationToken(), EventBus()
    bus.subscribe(
        lambda event: token.request()
        if event.kind is EventKind.CHECKPOINT_COMMITTED else None
    )
    store = HDF5ResultStore(tmp_path/"interrupted.h5")
    interrupted = SimulationSession(
        model=TimeModel(), initial_state=State(x=np.array([0.0])),
        schedule=schedule, nonlinear_solver=NonLinearSolver(),
        run_id="run_interrupted", result_store=store,
        checkpoint_every_report=True, checkpoint_schedule=schedule,
        code_fingerprint="code_test", streaming_execution=True,
        cancellation=token, event_bus=bus,
    ).run()
    assert interrupted.status is SimulationStatus.CANCELLED
    assert store.state_indices("run_interrupted") == (0,)
    assert store.inspect_run("run_interrupted")["status"] == "cancelled"
    np.testing.assert_array_equal(
        store.load_state("run_interrupted", 0)["x"], [1.0],
    )

    resumed = SimulationSession.from_checkpoint(
        source_store=store, source_run_id="run_interrupted",
        checkpoint_id="checkpoint_00000001", model=TimeModel(),
        full_schedule=schedule, nonlinear_solver=NonLinearSolver(),
        code_fingerprint="code_test", state_factory=State,
        run_id="run_after_interruption", result_store=store,
        checkpoint_every_report=True, streaming_execution=True,
    ).run()
    np.testing.assert_array_equal(
        resumed.final_state["x"], uninterrupted.final_state["x"],
    )
    assert resumed.final_state["x"][0] == 10.0


def test_abrupt_process_style_interruption_preserves_last_checkpoint(tmp_path):
    class TimeModel:
        @staticmethod
        def get_equations(state0, state, dt):
            (x,) = init_variables_adi(state["x"])
            return [x - state0["x"] - dt]

        @staticmethod
        def get_convergence(equations, dt):
            residual = float(np.max(np.abs(value(equations[0]))))
            return residual < 1e-12, {"residual": residual}

        @staticmethod
        def update_state(state, increment):
            updated = state.copy()
            updated["x"] = state["x"] + increment
            return updated

    schedule = np.array([1.0, 2.0, 3.0])
    store = HDF5ResultStore(tmp_path/"abrupt.h5")
    bus = EventBus()

    def terminate_process(event):
        if event.kind is EventKind.CHECKPOINT_COMMITTED:
            raise KeyboardInterrupt("simulated process termination")

    bus.subscribe(terminate_process)
    with pytest.raises(KeyboardInterrupt, match="process termination"):
        SimulationSession(
            model=TimeModel(), initial_state=State(x=np.array([0.0])),
            schedule=schedule, nonlinear_solver=NonLinearSolver(),
            run_id="run_abrupt", result_store=store,
            checkpoint_every_report=True, code_fingerprint="code_test",
            streaming_execution=True, event_bus=bus,
        ).run()

    source = store.inspect_run("run_abrupt")
    assert not source["complete"] and source["status"] == "running"
    assert source["accepted_states"] == 1 and source["checkpoints"] == 1
    record = store.load_checkpoint("run_abrupt", "checkpoint_00000001")
    np.testing.assert_array_equal(record["state"]["x"], [1.0])

    resumed = SimulationSession.from_checkpoint(
        source_store=store, source_run_id="run_abrupt",
        checkpoint_id="checkpoint_00000001", model=TimeModel(),
        full_schedule=schedule, nonlinear_solver=NonLinearSolver(),
        code_fingerprint="code_test", state_factory=State,
        run_id="run_recovered", result_store=store,
        streaming_execution=True,
    ).run()
    assert resumed.final_state["x"][0] == 6.0


def test_the_example_wrapper_is_observable_through_the_solver_it_is_given():
    """A thin example wrapper needs no hook of its own.

    `ad_blackoil.examples.utils.run_schedule` is one of the eleven runtime
    entry points and carries no `observer` parameter, which reads in the
    ledger as uninstrumented. It is not: it accepts a `solver` and delegates
    every step to `solver.solve_timestep`, so an instrumented `NonLinearSolver`
    observes the whole schedule through it. That distinction matters because
    the alternative - adding a hook to each wrapper - would duplicate the one
    the solver already owns and give two places for it to drift.

    The covenant half is asserted too: observation must not perturb the
    numbers, so the observed run has to be bit-identical, not merely close.
    """
    from collections import Counter
    from pymrst.modules.ad_blackoil.examples.blackoil_tutorial_one_phase import (
        build_blackoil_tutorial_one_phase,
    )
    from pymrst.modules.ad_blackoil.examples.utils.schedule import run_schedule

    model, state, timesteps, bc = build_blackoil_tutorial_one_phase()
    plain = run_schedule(model, state, timesteps[:3], controls={"bc": bc})

    seen = Counter()
    observed_model, observed_state, observed_steps, observed_bc = (
        build_blackoil_tutorial_one_phase()
    )
    observed = run_schedule(
        observed_model, observed_state, observed_steps[:3],
        controls={"bc": observed_bc},
        solver=NonLinearSolver(max_iterations=25,
                               observer=lambda event, payload: seen.update([event])),
    )

    assert set(seen) >= {"ministep_started", "nonlinear_iteration_completed",
                         "linear_solve_completed"}, dict(seen)
    assert sum(seen.values()) > 0
    np.testing.assert_array_equal(
        np.asarray(plain.states[-1]["pressure"], dtype=float),
        np.asarray(observed.states[-1]["pressure"], dtype=float),
    )
    assert ([step.iterations for step in plain.report.steps]
            == [step.iterations for step in observed.report.steps])


def test_the_compositional_schedule_loop_reports_its_own_decisions():
    """The wrapper that cannot be covered by delegation.

    `compositional._shared.run_schedule` drives `model.step` and owns its own
    timestep cutting and retry, so unlike its ad_blackoil sibling it takes no
    solver and the canonical hook never reaches it - a schedule run here
    produced no events at all. It now carries the same dependency-free hook,
    with the event names `NonLinearSolver` and the fixed-stress loop use.

    Per-iteration residuals stay out of reach because they live inside
    `model.step`; what this loop owns, and now reports, is its ministep and
    cut decisions.

    Observation must remain free: identical numbers with a recorder attached,
    and identical again under an observer that raises on every call.
    """
    from collections import Counter
    from pymrst.modules.compositional.examples.compositional_validation_simple import (
        setup_compositional_validation_simple_problem,
    )
    from pymrst.modules.compositional.examples._shared import run_schedule
    from pymrst.utils.units import DAY

    def go(observer):
        setup = setup_compositional_validation_simple_problem(
            use_natural=True, nstep=2, dt=10.0*DAY,
        )
        return run_schedule(setup["setup"].model, setup["setup"].state,
                            setup["schedule"], observer=observer)

    plain = go(None)

    seen = Counter()
    watched = go(lambda event, payload: seen.update([event]))
    assert set(seen) == {"report_step_started", "ministep_started", "step_accepted"}, dict(seen)

    def hostile(event, payload):
        raise RuntimeError("an observer must never reach the numerics")

    thrown = go(hostile)

    reference = np.asarray(plain.states[-1].pressure, dtype=float)
    for label, result in (("observed", watched), ("hostile", thrown)):
        np.testing.assert_array_equal(
            reference, np.asarray(result.states[-1].pressure, dtype=float),
            err_msg=f"{label} run perturbed the answer",
        )
        assert list(plain.iterations) == list(result.iterations), label


def test_a_real_coupled_model_through_the_session_matches_the_native_run():
    """Numerical characterization on a reservoir model, not a scalar one.

    The session's existing characterization test solves `x**2 - 4 = 0`. That
    establishes the delegation contract and nothing about a model with a
    coupled mechanics block, well unknowns and state fields the session must
    carry through untouched - which is what it will actually be asked to run.

    This drives MRST's 2D poromechanics case (80 cells, 210 displacement
    degrees of freedom, fully coupled) natively and through the session, and
    requires every state field to be bit-identical. Equality is exact on
    purpose: the session delegates to the same solver, so it is the same
    arithmetic in the same order, and any difference at all means the session
    perturbed something.
    """
    from pymrst.modules.ad_mechanics.examples.run_2d_case import run_2d_case

    case = run_2d_case(fluid_model="water", method="fully_coupled",
                       nx=20, ny=4, nsteps=3)
    schedule = np.asarray(case.dts, dtype=float)
    initial = case.states[0]

    native_states, native_report = NonLinearSolver(
        max_iterations=100).simulate_schedule(case.model, initial, schedule)
    session = SimulationSession(
        model=case.model, initial_state=initial, schedule=schedule,
        nonlinear_solver=NonLinearSolver(max_iterations=100),
        run_id="run_real_coupled_model",
    ).run()

    assert len(session.states) == len(native_states) == len(schedule)
    fields = sorted(native_states[-1].keys())
    assert {"pressure", "u", "xd"} <= set(fields), fields
    for index, (native, observed) in enumerate(zip(native_states, session.states)):
        for field in fields:
            np.testing.assert_array_equal(
                np.asarray(native[field], dtype=float),
                np.asarray(observed[field], dtype=float),
                err_msg=f"state {index} field {field!r} differs",
            )
    assert ([step.iterations for step in native_report.steps]
            == [step.iterations for step in session.report.steps])
    assert ([step.cuts for step in native_report.steps]
            == [step.cuts for step in session.report.steps])
    assert session.evidence.to_json()
