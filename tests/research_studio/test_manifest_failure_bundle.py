"""Manifest lineage and independently loadable failure-bundle tests."""

import json
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pytest

from pymrst.research_studio import (
    CaseId, CheckpointRecord, RunId, RunIdentity, SimulationEvidence,
    capture_manifest, load_failure_bundle, prepare_failure_replay,
    write_failure_bundle,
)
from pymrst.research_studio.runtime.identifiers import CheckpointId
from pymrst.research_studio.storage.checkpoint import (
    default_model_fingerprint, default_solver_fingerprint,
    schedule_fingerprint,
)


def test_manifest_captures_and_verifies_input_lineage(tmp_path):
    deck = tmp_path/"CASE.DATA"
    deck.write_text("RUNSPEC\n", encoding="utf-8")
    manifest = capture_manifest(
        "run_manifest", code_fingerprint="code_abc", inputs=[deck],
        command=["python3", "case.py"], configuration={"tolerance": 1e-8},
    )
    manifest.verify_inputs()
    assert manifest.inputs[0].size_bytes == len("RUNSPEC\n")
    assert manifest.package_versions["numpy"] != "not-installed"
    restored = type(manifest).from_dict(json.loads(manifest.to_json()))
    assert restored.to_dict() == manifest.to_dict()
    deck.write_text("CHANGED\n", encoding="utf-8")
    with pytest.raises(ValueError, match="changed"):
        manifest.verify_inputs()


def _bundle_inputs(tmp_path):
    source = tmp_path/"input.txt"
    source.write_text("input", encoding="utf-8")
    manifest = capture_manifest(
        "run_failure", code_fingerprint="code_test", inputs=[source],
        command=["python3", "replay.py"],
    )
    evidence = SimulationEvidence(
        RunIdentity(RunId("run_failure"), CaseId("case_failure")),
        failures=[{"type": "NumericalFailureAnalysis"}],
    )
    checkpoint = CheckpointRecord(
        checkpoint_id=CheckpointId("checkpoint_00000003"),
        source_run_id=RunId("run_failure"), completed_report_steps=3,
        accepted_elapsed_time=6.0,
        state={"pressure": np.array([1.0e7, 0.9e7]), "wellSol": [{"bhp": 8e6}]},
        schedule_fingerprint="schedule", model_fingerprint="model",
        solver_fingerprint="solver", code_fingerprint="code_test",
    )
    failing_step = {
        "dt": 4.0, "report_step": 3,
        "control": {"name": "production"},
        "initial_guess": np.array([1.0, 2.0]),
    }
    return manifest, evidence, checkpoint, failing_step


def test_failure_bundle_roundtrip_uses_nonpickled_arrays(tmp_path):
    manifest, evidence, checkpoint, failing_step = _bundle_inputs(tmp_path)
    path = write_failure_bundle(
        tmp_path/"failure.pymrst-failure", manifest=manifest,
        evidence=evidence, checkpoint=checkpoint, failing_step=failing_step,
    )
    loaded = load_failure_bundle(path)
    np.testing.assert_array_equal(loaded["checkpoint"].state["pressure"], checkpoint.state["pressure"])
    np.testing.assert_array_equal(loaded["failing_step"]["initial_guess"], [1.0, 2.0])
    assert loaded["manifest"].run_id == manifest.run_id
    assert loaded["evidence"].failures == evidence.failures
    with ZipFile(path) as archive:
        assert all(not name.endswith((".pkl", ".pickle")) for name in archive.namelist())
        assert any(name.endswith(".npy") for name in archive.namelist())


def test_failure_bundle_detects_member_corruption(tmp_path):
    manifest, evidence, checkpoint, failing_step = _bundle_inputs(tmp_path)
    path = write_failure_bundle(
        tmp_path/"failure.zip", manifest=manifest, evidence=evidence,
        checkpoint=checkpoint, failing_step=failing_step,
    )
    with pytest.warns(UserWarning, match="Duplicate name"):
        with ZipFile(path, "a", compression=ZIP_DEFLATED) as archive:
            archive.writestr("bundle.json", b"{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_failure_bundle(path)


def test_failure_bundle_refuses_silent_overwrite(tmp_path):
    manifest, evidence, checkpoint, failing_step = _bundle_inputs(tmp_path)
    path = tmp_path/"failure.zip"
    write_failure_bundle(
        path, manifest=manifest, evidence=evidence,
        checkpoint=checkpoint, failing_step=failing_step,
    )
    with pytest.raises(FileExistsError):
        write_failure_bundle(
            path, manifest=manifest, evidence=evidence,
            checkpoint=checkpoint, failing_step=failing_step,
        )


def test_failure_replay_validates_then_runs_native_solver(tmp_path):
    class ReplayModel:
        pass

    class ReplaySolver:
        def simulate_schedule(self, model, state, schedule):
            return [
                {"pressure": state["pressure"] + float(dt)} for dt in schedule
            ], {"replayed": True}

    model, solver = ReplayModel(), ReplaySolver()
    schedule = np.array([1.0, 2.0, 3.0])
    source = tmp_path/"CASE.DATA"
    source.write_text("RUNSPEC\n", encoding="utf-8")
    manifest = capture_manifest(
        "run_replay", code_fingerprint="code_replay", inputs=[source],
    )
    evidence = SimulationEvidence(
        RunIdentity(RunId("run_replay"), CaseId("case_replay")),
    )
    checkpoint = CheckpointRecord(
        checkpoint_id=CheckpointId("checkpoint_00000001"),
        source_run_id=RunId("run_replay"), completed_report_steps=1,
        accepted_elapsed_time=1.0, state={"pressure": np.array([10.0])},
        schedule_fingerprint=schedule_fingerprint(schedule),
        model_fingerprint=default_model_fingerprint(model),
        solver_fingerprint=default_solver_fingerprint(solver),
        code_fingerprint="code_replay",
    )
    path = write_failure_bundle(
        tmp_path/"replay.pymrst-failure", manifest=manifest,
        evidence=evidence, checkpoint=checkpoint,
        failing_step={"report_step": 1, "reason": "nonlinear_failure"},
    )
    replay = prepare_failure_replay(
        path, model=model, nonlinear_solver=solver, full_schedule=schedule,
        code_fingerprint="code_replay",
    )
    np.testing.assert_array_equal(replay.remaining_schedule, [2.0, 3.0])
    states, report = replay.run()
    np.testing.assert_array_equal(states[0]["pressure"], [12.0])
    assert report == {"replayed": True}
    assert replay.failing_step["reason"] == "nonlinear_failure"


def test_failure_replay_rejects_code_or_numerical_owner_mismatch(tmp_path):
    class Model:
        pass

    class Solver:
        def simulate_schedule(self, model, state, schedule):
            return [], None

    model, solver = Model(), Solver()
    schedule = np.array([1.0])
    source = tmp_path/"input.DATA"
    source.write_text("DATA", encoding="utf-8")
    manifest = capture_manifest("run", code_fingerprint="code", inputs=[source])
    checkpoint = CheckpointRecord(
        CheckpointId("checkpoint_00000000"), RunId("run"), 0, 0.0,
        {"pressure": np.array([1.0])}, schedule_fingerprint(schedule),
        default_model_fingerprint(model), default_solver_fingerprint(solver), "code",
    )
    path = write_failure_bundle(
        tmp_path/"failure.zip", manifest=manifest,
        evidence=SimulationEvidence(RunIdentity(RunId("run"), CaseId("case"))),
        checkpoint=checkpoint, failing_step={},
    )
    with pytest.raises(ValueError, match="manifest code"):
        prepare_failure_replay(
            path, model=model, nonlinear_solver=solver, full_schedule=schedule,
            code_fingerprint="changed",
        )

    class ChangedSolver(Solver):
        pass

    with pytest.raises(RuntimeError, match="solver"):
        prepare_failure_replay(
            path, model=model, nonlinear_solver=ChangedSolver(),
            full_schedule=schedule, code_fingerprint="code",
        )
