from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import pytest

from pymrst.app.database.models import AppDatabase
from pymrst.app.schemas.workload import (
    WorkloadBudget, WorkloadRequest, WorkloadStatus, WorkloadTier,
)
from pymrst.app.services.workload_service import WorkloadQualificationService


def _wait(service, identifier, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        record = service.get(identifier)
        if record.status not in {WorkloadStatus.QUEUED, WorkloadStatus.RUNNING}:
            return record
        time.sleep(0.03)
    raise AssertionError("workload qualification did not finish")


def test_smoke_workload_retains_budgeted_evidence_and_baseline(tmp_path):
    database = AppDatabase(tmp_path / ".pymrst-app" / "metadata.sqlite")
    service = WorkloadQualificationService(database, tmp_path)
    try:
        first = _wait(service, service.submit(
            WorkloadRequest("numerical-core", "smoke", 42),
        ).qualification_id)
        assert first.status is WorkloadStatus.PASSED
        assert first.metrics["cells"] == 5000
        assert first.metrics["linear_residual_inf"] < 1e-10
        assert first.metrics["warmup_runs"] == 1
        assert first.metrics["measured_runs"] == 3
        assert first.metrics["statistics"]["wall_seconds"]["count"] == 3
        assert first.metrics["wall_seconds"] == first.metrics["statistics"]["wall_seconds"]["median"]
        assert first.metrics["correctness"]["deterministic_across_measured_runs"] is True
        assert first.metrics["correctness"]["resource_telemetry_collected"] is True
        assert first.metrics["telemetry"]["sample_count"] >= 2
        assert "samples" not in first.metrics["telemetry"]
        evidence = tmp_path / first.evidence_relative_path
        telemetry_path = evidence.parent / first.metrics["telemetry"]["artifact_relative_path"]
        telemetry_document = json.loads(telemetry_path.read_text())
        assert hashlib.sha256(telemetry_path.read_bytes()).hexdigest() == first.metrics["telemetry"]["artifact_sha256"]
        telemetry_times = [sample["relative_seconds"]
                           for sample in telemetry_document["samples"]]
        assert telemetry_times == sorted(telemetry_times)
        assert first.metrics["telemetry"]["summary"]["sampled_peak_rss_bytes"] > 0
        assert first.comparison["method"] == "median_of_measured_runs"
        assert first.comparison["baseline_qualification_id"] == ""
        assert evidence.is_file()
        assert hashlib.sha256(evidence.read_bytes()).hexdigest() == first.content_hash
        assert json.loads(evidence.read_text())["metrics"]["result_checksum"] == first.metrics["result_checksum"]
        assert service.verify_evidence(first.qualification_id)["valid"] is True
        evidence.write_text("tampered\n")
        assert service.verify_evidence(first.qualification_id)["valid"] is False

        second = _wait(service, service.submit(
            WorkloadRequest("numerical-core", "smoke", 42),
        ).qualification_id)
        assert second.comparison["baseline_qualification_id"] == first.qualification_id
        assert set(second.comparison["ratios"]) == {"wall_seconds", "peak_rss_bytes"}
    finally:
        service.shutdown(); database.close()


def test_absolute_budget_failure_is_evidence_not_worker_failure(tmp_path):
    database = AppDatabase(tmp_path / "metadata.sqlite")
    service = WorkloadQualificationService(database, tmp_path)
    budget = WorkloadBudget(
        maximum_wall_seconds=1e-9, maximum_cpu_seconds=30,
        maximum_peak_rss_bytes=2 * 1024**3, maximum_disk_bytes=1024**3,
    )
    try:
        record = _wait(service, service.submit(
            WorkloadRequest("numerical-core", "smoke", 7, budget),
        ).qualification_id)
        assert record.status is WorkloadStatus.FAILED
        assert record.metrics["result_checksum"]
        assert record.comparison["checks"]["wall_seconds"]["passed"] is False
        assert record.error["error_type"] == "PerformanceBudgetExceeded"
        assert (tmp_path / record.evidence_relative_path).is_file()
    finally:
        service.shutdown(); database.close()


def test_correctness_invariant_fails_independently_of_resource_budgets():
    budget = WorkloadBudget(60, 60, 1024**3, 1024**3)
    passed, comparison = WorkloadQualificationService._assess({
        "wall_seconds": 1.0, "cpu_seconds": 0.8,
        "peak_rss_bytes": 1024, "io_bytes": 0,
        "correctness": {"finite_state": True, "force_balance": False},
    }, budget, None)
    assert passed is False
    assert all(check["passed"] for check in comparison["checks"].values())
    assert comparison["correctness"] == {
        "finite_state": True, "force_balance": False,
    }


def test_excessive_timing_dispersion_fails_even_when_absolute_time_passes():
    budget = WorkloadBudget(
        60, 60, 1024**3, 1024**3, maximum_wall_relative_mad=0.10,
    )
    passed, comparison = WorkloadQualificationService._assess({
        "wall_seconds": 10.0, "cpu_seconds": 8.0,
        "peak_rss_bytes": 1024, "io_bytes": 0,
        "statistics": {"wall_seconds": {"median": 10.0, "mad": 2.0}},
    }, budget, None)
    assert passed is False
    assert comparison["checks"]["wall_relative_mad"] == {
        "actual": 0.2, "maximum": 0.1, "passed": False,
    }


def test_repetition_engine_rejects_nondeterministic_result_hashes(tmp_path, monkeypatch):
    from pymrst.app.workers import workload_entry

    hashes = iter(("warmup", "a", "b", "a"))
    monkeypatch.setattr(workload_entry, "_dispatch", lambda *_args: {
        "operations": 1, "result_checksum": next(hashes),
        "correctness": {"finite": True},
    })
    metrics = workload_entry._run_repeated({
        "workload_id": "manufactured", "tier": "smoke", "seed": 1,
        "warmup_runs": 1, "measured_runs": 3,
    }, tmp_path)
    assert metrics["measured_result_checksums"] == ["a", "b", "a"]
    assert metrics["correctness"]["deterministic_across_measured_runs"] is False


def test_endurance_workload_can_be_cancelled_without_orphaning_process(tmp_path):
    database = AppDatabase(tmp_path / "metadata.sqlite")
    service = WorkloadQualificationService(database, tmp_path)
    try:
        submitted = service.submit(WorkloadRequest("numerical-core", "endurance", 9))
        service.cancel(submitted.qualification_id)
        record = _wait(service, submitted.qualification_id)
        assert record.status is WorkloadStatus.CANCELLED
        assert record.finished_at
        assert record.cancellation_requested
    finally:
        service.shutdown(); database.close()


def test_schema_19_migrates_to_workload_registry(tmp_path):
    path = tmp_path / "metadata.sqlite"
    database = AppDatabase(path); database.close()
    connection = sqlite3.connect(path)
    connection.execute("DROP TABLE app_workload_qualifications")
    connection.execute("PRAGMA user_version=19")
    connection.commit(); connection.close()
    migrated = AppDatabase(path)
    assert migrated.schema_version == 20
    assert migrated.fetch_one(
        "SELECT name FROM sqlite_master WHERE name='app_workload_qualifications'"
    ) is not None
    migrated.close()


def test_workload_requests_and_history_are_bounded(tmp_path):
    with pytest.raises(ValueError, match="seed"):
        WorkloadRequest("numerical-core", "smoke", -1)
    with pytest.raises(ValueError, match="odd integer"):
        WorkloadRequest("numerical-core", "smoke", measured_runs=2)
    with pytest.raises(ValueError, match="warmup_runs"):
        WorkloadRequest("numerical-core", "smoke", warmup_runs=6)
    database = AppDatabase(tmp_path / "metadata.sqlite")
    service = WorkloadQualificationService(database, tmp_path)
    with pytest.raises(ValueError, match="between 1 and 1000"):
        service.list(limit=0)
    with pytest.raises(ValueError, match="unknown workload"):
        service.submit(WorkloadRequest("not-registered", "smoke"))
    service.shutdown(); database.close()


def test_catalog_reports_dataset_availability_without_accepting_arbitrary_paths(
    tmp_path, monkeypatch,
):
    monkeypatch.delenv("PYMRST_SPE1_DECK", raising=False)
    database = AppDatabase(tmp_path / "metadata.sqlite")
    service = WorkloadQualificationService(database, tmp_path)
    try:
        definition = {item.workload_id: item for item in service.catalog()}["spe1-blackoil"]
        assert definition.tiers == (WorkloadTier.SMOKE, WorkloadTier.ENGINEERING)
        assert definition.requirements[0] == "BENCH_SPE1.DATA"
        assert definition.available is False
        with pytest.raises(ValueError, match="BENCH_SPE1.DATA"):
            service.submit(WorkloadRequest("spe1-blackoil", "smoke", 1))
    finally:
        service.shutdown(); database.close()


def test_real_spe1_smoke_workload_certifies_solver_and_state(tmp_path, monkeypatch):
    deck = (Path(__file__).resolve().parents[3] / "mrst-2024a-Geomech" /
            "examples" / "data" / "SPE1" / "BENCH_SPE1.DATA")
    if not deck.is_file():
        pytest.skip("SPE1 dataset is not installed beside this checkout")
    monkeypatch.setenv("PYMRST_SPE1_DECK", str(deck))
    database = AppDatabase(tmp_path / "metadata.sqlite")
    service = WorkloadQualificationService(database, tmp_path)
    try:
        definition = {item.workload_id: item for item in service.catalog()}["spe1-blackoil"]
        assert definition.available is True
        record = _wait(service, service.submit(
            WorkloadRequest("spe1-blackoil", "smoke", 2026),
        ).qualification_id, timeout=90)
        assert record.status is WorkloadStatus.PASSED, record.error
        assert record.metrics["cells"] == 300
        assert record.metrics["report_steps"] == 1
        assert all(record.metrics["correctness"].values())
        assert all(record.comparison["correctness"].values())
        assert record.metrics["deck_sha256"] == hashlib.sha256(deck.read_bytes()).hexdigest()
        evidence = tmp_path / record.evidence_relative_path
        telemetry_path = evidence.parent / record.metrics["telemetry"]["artifact_relative_path"]
        telemetry_document = json.loads(telemetry_path.read_text())
        assert any(sample["phase"] == "measured"
                   for sample in telemetry_document["samples"])
    finally:
        service.shutdown(); database.close()


def test_mandel_smoke_workload_certifies_coupling_and_analytic_response(tmp_path):
    database = AppDatabase(tmp_path / "metadata.sqlite")
    service = WorkloadQualificationService(database, tmp_path)
    try:
        record = _wait(service, service.submit(
            WorkloadRequest("mandel-poromechanics", "smoke", 2026),
        ).qualification_id, timeout=90)
        assert record.status is WorkloadStatus.PASSED, record.error
        assert record.metrics["cells"] == 48
        assert record.metrics["accepted_coupled_steps"] == 20
        assert record.metrics["centre_peak_pressure_ratio"] > 1.05
        assert record.metrics["maximum_pressure_relative_error"] < 0.06
        assert record.metrics["final_plate_force_relative_error"] < 1e-6
        assert all(record.metrics["correctness"].values())
        assert all(record.comparison["correctness"].values())
        verification = service.verify_evidence(record.qualification_id)
        assert verification["valid"] is True
        assert verification["artifacts"][0]["valid"] is True
        evidence = tmp_path / record.evidence_relative_path
        telemetry_path = evidence.parent / record.metrics["telemetry"]["artifact_relative_path"]
        telemetry_path.write_text("tampered\n")
        assert service.verify_evidence(record.qualification_id)["valid"] is False
    finally:
        service.shutdown(); database.close()


def test_workload_http_surface_submits_and_reports_evidence(tmp_path):
    fastapi = pytest.importorskip("fastapi")
    assert fastapi
    from fastapi.testclient import TestClient
    from pymrst.app.application import create_application
    from pymrst.app.api.application import create_fastapi_app

    context = create_application(tmp_path)
    try:
        with TestClient(create_fastapi_app(context)) as client:
            catalog = client.get("/api/v1/system/workload-catalog")
            assert catalog.status_code == 200
            assert {item["workload_id"] for item in catalog.json()} == {
                "numerical-core", "state-io", "spe1-blackoil",
                "mandel-poromechanics",
            }
            response = client.post("/api/v1/system/workload-qualifications", json={
                "workload_id": "state-io", "tier": "smoke", "seed": 123,
                "budget": None,
            })
            assert response.status_code == 202
            identifier = response.json()["qualification_id"]
            deadline = time.monotonic() + 15
            while time.monotonic() < deadline:
                response = client.get(
                    f"/api/v1/system/workload-qualifications/{identifier}"
                )
                assert response.status_code == 200
                if response.json()["status"] not in {"queued", "running"}:
                    break
                time.sleep(0.03)
            result = response.json()
            assert result["status"] == "passed"
            assert result["metrics"]["io_bytes"] > 0
            assert result["content_hash"]
    finally:
        context.close()
