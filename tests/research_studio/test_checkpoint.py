"""Checkpoint v1 compatibility and restart-equivalence tests."""

import numpy as np
import pytest

from pymrst.modules.ad_core.simulators.nonlinear_solver import NonLinearSolver
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.research_studio import CheckpointCompatibilityError, CheckpointRecord
from pymrst.research_studio.runtime.identifiers import CheckpointId, RunId
from pymrst.research_studio.storage.checkpoint import (
    default_model_fingerprint, default_solver_fingerprint, restart_payload,
    schedule_fingerprint,
)
from pymrst.utils.adi import init_variables_adi, value


class AccumulatingModel:
    """x advances by dt, making restart state and schedule position observable."""

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


def _record(state, schedule, model, solver):
    return CheckpointRecord(
        checkpoint_id=CheckpointId("checkpoint_2"),
        source_run_id=RunId("run_source"),
        completed_report_steps=2,
        accepted_elapsed_time=float(np.sum(schedule[:2])),
        state=state,
        schedule_fingerprint=schedule_fingerprint(schedule),
        model_fingerprint=default_model_fingerprint(model),
        solver_fingerprint=default_solver_fingerprint(solver),
        code_fingerprint="code_test",
    )


def test_resumed_result_matches_uninterrupted_schedule_exactly():
    model, solver = AccumulatingModel(), NonLinearSolver()
    schedule = np.array([1.0, 2.0, 3.0, 4.0])
    initial = State(x=np.array([0.0]))
    uninterrupted, _ = solver.simulate_schedule(model, initial, schedule)
    prefix, _ = NonLinearSolver().simulate_schedule(model, initial, schedule[:2])
    checkpoint = _record(prefix[-1], schedule, model, solver)
    restored, remaining = restart_payload(
        checkpoint,
        schedule=schedule,
        model_fingerprint=default_model_fingerprint(model),
        solver_fingerprint=default_solver_fingerprint(solver),
        code_fingerprint="code_test",
        state_factory=State,
    )
    resumed, _ = NonLinearSolver().simulate_schedule(model, restored, remaining)
    np.testing.assert_array_equal(resumed[-1]["x"], uninterrupted[-1]["x"])
    np.testing.assert_array_equal(remaining, schedule[2:])


def test_changed_schedule_solver_model_or_code_is_rejected():
    model, solver = AccumulatingModel(), NonLinearSolver()
    schedule = np.array([1.0, 2.0, 3.0])
    checkpoint = _record(State(x=np.array([3.0])), schedule, model, solver)
    variants = [
        dict(schedule=[1.0, 2.0, 4.0], model_fingerprint=checkpoint.model_fingerprint,
             solver_fingerprint=checkpoint.solver_fingerprint, code_fingerprint="code_test"),
        dict(schedule=schedule, model_fingerprint="different",
             solver_fingerprint=checkpoint.solver_fingerprint, code_fingerprint="code_test"),
        dict(schedule=schedule, model_fingerprint=checkpoint.model_fingerprint,
             solver_fingerprint="different", code_fingerprint="code_test"),
        dict(schedule=schedule, model_fingerprint=checkpoint.model_fingerprint,
             solver_fingerprint=checkpoint.solver_fingerprint, code_fingerprint="different"),
    ]
    for variant in variants:
        with pytest.raises(CheckpointCompatibilityError, match="mismatch"):
            restart_payload(checkpoint, **variant)


def test_checkpoint_schema_roundtrip_and_structured_schedule_fingerprint():
    from pymrst.modules.ad_core.utils.schedule import simple_schedule

    model, solver = AccumulatingModel(), NonLinearSolver()
    schedule = np.array([1.0, 2.0, 3.0])
    record = _record(State(x=np.array([3.0])), schedule, model, solver)
    restored = CheckpointRecord.from_dict(record.to_dict())
    assert restored.completed_report_steps == record.completed_report_steps
    np.testing.assert_array_equal(restored.state["x"], record.state["x"])
    structured = simple_schedule(schedule)
    assert schedule_fingerprint(structured) == schedule_fingerprint(simple_schedule(schedule))
    assert schedule_fingerprint(structured) != schedule_fingerprint(schedule)
