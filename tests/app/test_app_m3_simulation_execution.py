"""Acceptance tests for isolated App M3 run orchestration."""

from __future__ import annotations

from pathlib import Path
import time

import pytest

from pymrst.app import create_application
from pymrst.app.database import APP_DATABASE_SCHEMA_VERSION
from pymrst.app.schemas import ProjectCreate, RunStatus, TaskStatus


def wait_for_run(app, run_id, statuses=None, timeout=10):
    statuses = set(statuses or {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED})
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = app.runs.get(run_id)
        if run.status in statuses:
            return run
        time.sleep(0.02)
    raise TimeoutError(run_id)


def make_configuration(app, project_id, payload):
    return app.runs.create_configuration(
        project_id, "Protocol qualification", "protocol-qualification", payload,
    )


def test_isolated_run_persists_progress_events_result_and_process_evidence(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Execution"))
        configuration = make_configuration(app, project.project_id, {"steps": 4})
        submitted = app.isolated.submit(configuration.configuration_id)
        completed = wait_for_run(app, submitted.run_id)
        task = app.tasks.get(completed.task_id)
        events = app.runs.events(completed.run_id)
        assert completed.status is RunStatus.COMPLETED
        assert completed.process_id and completed.process_id != 0
        assert task.status is TaskStatus.COMPLETED
        assert task.result == {"report_steps": 4, "resumed_from_step": 0}
        assert [item.payload["report_step"] for item in events
                if item.event_type == "progress" and "report_step" in item.payload] == [1, 2, 3, 4]
        assert len([item for item in events if item.event_type == "convergence"]) == 4
        assert [item.sequence for item in events] == list(range(len(events)))
        run_root = Path(project.root_path) / completed.run_relative_path
        assert (run_root / "manifest.json").is_file()
        assert (run_root / "status.json").is_file()
        assert len((run_root / "events.jsonl").read_text().splitlines()) == len(events)
        assert (run_root / "worker_request.json").is_file()
        assert (run_root / "stderr.log").is_file()


def test_failure_is_isolated_and_retains_structured_traceback(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Failure"))
        configuration = make_configuration(app, project.project_id, {"steps": 1, "fail": True})
        run = wait_for_run(app, app.isolated.submit(configuration.configuration_id).run_id)
        task = app.tasks.get(run.task_id)
        assert run.status is RunStatus.FAILED
        assert task.status is TaskStatus.FAILED
        assert task.error["error_type"] == "RuntimeError"
        assert "qualification failure" in task.error["message"]
        failure = next(item for item in app.runs.events(run.run_id) if item.event_type == "failure")
        assert "protocol_jobs.py" in failure.payload["traceback"]


def test_checkpoint_resume_creates_independent_attempt_and_lineage(tmp_path):
    root = tmp_path / "workspace"
    with create_application(root) as app:
        project = app.projects.create(ProjectCreate("Resume"))
        configuration = make_configuration(app, project.project_id, {"steps": 5, "checkpoint_at": 2})
        parent = wait_for_run(app, app.isolated.submit(configuration.configuration_id).run_id)
        checkpoint = app.runs.checkpoints(parent.run_id)[0]
        child = wait_for_run(app, app.isolated.resume(parent.run_id, checkpoint.checkpoint_id).run_id)
        assert child.status is RunStatus.COMPLETED
        assert child.parent_run_id == parent.run_id
        assert child.resume_checkpoint_id == checkpoint.checkpoint_id
        assert child.attempt_number == 2
        assert app.tasks.get(child.task_id).result["resumed_from_step"] == 2
        assert app.runs.checkpoint_path(checkpoint.checkpoint_id).is_file()
        assert parent.run_relative_path != child.run_relative_path

    with create_application(root) as reopened:
        restored = reopened.runs.get(child.run_id)
        assert restored == child
        assert len(reopened.runs.events(parent.run_id)) > 0
        assert reopened.runs.checkpoints(parent.run_id)[0] == checkpoint


def test_running_child_can_be_cancelled_without_stopping_application(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Cancellation"))
        configuration = make_configuration(
            app, project.project_id, {"steps": 100, "delay_seconds": 0.03},
        )
        submitted = app.isolated.submit(configuration.configuration_id)
        wait_for_run(app, submitted.run_id, {RunStatus.RUNNING}, timeout=5)
        app.isolated.cancel(submitted.run_id)
        cancelled = wait_for_run(app, submitted.run_id)
        assert cancelled.status is RunStatus.CANCELLED
        assert app.tasks.get(cancelled.task_id).status is TaskStatus.CANCELLED
        other = app.projects.create(ProjectCreate("Application survived"))
        assert app.projects.get(other.project_id) == other


def test_runner_registration_is_trusted_and_payload_cannot_select_entrypoint(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        with pytest.raises(ValueError, match="trusted PyMRST"):
            app.isolated.register("unsafe", "os:system")
        project = app.projects.create(ProjectCreate("Security"))
        configuration = make_configuration(app, project.project_id, {
            "steps": 1, "entrypoint": "os:system", "command": ["touch", "owned"],
        })
        run = wait_for_run(app, app.isolated.submit(configuration.configuration_id).run_id)
        assert run.status is RunStatus.COMPLETED
        assert not (Path(project.root_path) / run.run_relative_path / "owned").exists()


def test_app_m3_database_schema_is_present(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        assert app.database.schema_version == APP_DATABASE_SCHEMA_VERSION
        names = {row[0] for row in app.database.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert {
            "app_run_configurations", "app_runs", "app_run_events", "app_run_checkpoints",
        } <= names
