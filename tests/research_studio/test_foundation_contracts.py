"""Characterization tests for the first persistent Research Studio contracts."""

from dataclasses import FrozenInstanceError
from io import StringIO
import json
from uuid import UUID

import pytest

from pymrst.research_studio import (
    AttemptId, CaseId, EventBus, EventKind, JsonLinesHandler, RunId,
    RunIdentity, SimulationEvent, SimulationEvidence,
)


def test_identifiers_are_typed_stable_strings():
    value = RunId.from_uuid(UUID(int=1))
    assert value == "run_00000000000000000000000000000001"
    assert json.loads(json.dumps({"run": value})) == {"run": str(value)}
    assert RunId.generate().startswith("run_")
    with pytest.raises(ValueError):
        CaseId("contains whitespace")


def test_events_are_immutable_and_payload_cannot_be_changed_by_handler():
    event = SimulationEvent(
        EventKind.STEP_REJECTED, RunId("run_test"), 4, 2.5,
        attempt_id=AttemptId("attempt_2"),
        payload={"dt": 10.0, "residuals": [1.0, 0.1]},
    )
    with pytest.raises(TypeError):
        event.payload["dt"] = 5.0
    with pytest.raises(FrozenInstanceError):
        event.sequence = 5
    with pytest.raises(TypeError):
        event.payload["residuals"][0] = 9.0


def test_evidence_json_roundtrip_is_strict_and_versioned():
    evidence = SimulationEvidence(
        RunIdentity(RunId("run_test"), CaseId("case_water")),
        manifest={"command": "water_case"},
        convergence={"iterations": [2, 3]},
    )
    restored = SimulationEvidence.from_json(evidence.to_json())
    assert restored.to_dict() == evidence.to_dict()

    unsupported = evidence.to_dict()
    unsupported["schema_version"] = 2
    with pytest.raises(ValueError, match="unsupported"):
        SimulationEvidence.from_dict(unsupported)


def test_persistence_rejects_runtime_objects_and_nonfinite_numbers():
    evidence = SimulationEvidence(
        RunIdentity(RunId("run_test"), CaseId("case_water")),
        manifest={"model": object()},
    )
    with pytest.raises(TypeError, match="unsupported persistent value"):
        evidence.to_dict()
    evidence.manifest = {"residual": float("nan")}
    with pytest.raises(ValueError, match="non-finite"):
        evidence.to_json()


def test_event_bus_isolates_handlers_and_json_lines_is_complete():
    stream = StringIO()
    received = []
    bus = EventBus()
    bus.subscribe(lambda event: received.append(event.sequence))

    def broken(_event):
        raise RuntimeError("renderer unavailable")

    bus.subscribe(broken)
    bus.subscribe(JsonLinesHandler(stream))
    event = SimulationEvent(
        EventKind.RUN_STARTED, RunId("run_test"), 0, 0.0,
        payload={"nested": {"case": "water"}},
    )
    bus.publish(event)
    assert received == [0]
    assert len(bus.failures) == 1
    assert json.loads(stream.getvalue())["payload"]["nested"]["case"] == "water"
