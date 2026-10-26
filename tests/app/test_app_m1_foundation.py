"""Acceptance tests for the App M1 project, database, and task foundation."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

import pytest

from pymrst.app import create_application
from pymrst.app.database import APP_DATABASE_SCHEMA_VERSION, AppDatabase
from pymrst.app.schemas import ProjectCreate, TaskStatus, WorkspaceState


def wait_for_task(app, task_id, timeout=3):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = app.tasks.get(task_id)
        if task.status in {
            TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED,
        }:
            return task
        time.sleep(0.01)
    raise TimeoutError(task_id)


def test_project_creation_is_durable_atomic_and_has_professional_layout(tmp_path):
    root = tmp_path / "workspace"
    with create_application(root, background_workers=1) as app:
        project = app.projects.create(ProjectCreate(
            "North Sea Study", "Qualification project", ("black-oil", "research"),
        ))
        project_root = Path(project.root_path)
        assert project_root.parent == root / "projects"
        assert json.loads((project_root / "project.json").read_text())["project_id"] \
            == project.project_id
        for relative in (
            "inputs/original", "models", "variants", "runs", "reports",
            "experiments", "investigations", ".pymrst",
        ):
            assert (project_root / relative).is_dir()
        navigation = app.projects.navigation(project.project_id)
        assert [item.item_id for item in navigation.items] == [
            "overview", "model", "simulation", "results", "experiments",
            "agents", "reports",
        ]

    with create_application(root, background_workers=1) as reopened:
        restored = reopened.projects.get(project.project_id)
        assert restored == project
        assert reopened.database.schema_version == APP_DATABASE_SCHEMA_VERSION


def test_input_import_is_content_addressed_deduplicated_and_read_only(tmp_path):
    source = tmp_path / "SPE1.DATA"
    source.write_text("RUNSPEC\nDIMENS\n10 10 3 /\n")
    original = source.read_bytes()
    with create_application(tmp_path / "workspace", background_workers=1) as app:
        project = app.projects.create(ProjectCreate("SPE1"))
        first = app.projects.import_input(project.project_id, source, kind="deck")
        second = app.projects.import_input(project.project_id, source, kind="deck")
        assert first == second
        assert app.projects.assets(project.project_id) == (first,)
        imported = Path(project.root_path) / first.relative_path
        assert imported.read_bytes() == original
        assert not os.access(imported, os.W_OK) or imported.stat().st_mode & 0o222 == 0
        assert source.read_bytes() == original
        other = app.projects.create(ProjectCreate("SPE1 copy"))
        other_asset = app.projects.import_input(other.project_id, source, kind="deck")
        assert other_asset.content_hash == first.content_hash
        assert other_asset.asset_id != first.asset_id


def test_task_lifecycle_is_strict_and_restart_marks_running_tasks_interrupted(tmp_path):
    root = tmp_path / "workspace"
    app = create_application(root, background_workers=1)
    project = app.projects.create(ProjectCreate("Restart test"))
    task = app.tasks.create(project.project_id, "safe-index", {"asset": "a"})
    running = app.tasks.transition(
        task.task_id, TaskStatus.RUNNING, message="Indexing",
    )
    assert running.status is TaskStatus.RUNNING
    app.close()

    with create_application(root, background_workers=1) as reopened:
        interrupted = reopened.tasks.get(task.task_id)
        assert interrupted.status is TaskStatus.INTERRUPTED
        assert "restarted" in interrupted.message.lower()
        notifications = reopened.notifications.list(project_id=project.project_id)
        assert len(notifications) == 1
        assert notifications[0].task_id == task.task_id
        with pytest.raises(ValueError, match="illegal task transition"):
            reopened.tasks.transition(task.task_id, TaskStatus.COMPLETED)


def test_safe_background_tasks_report_progress_result_and_notification(tmp_path):
    with create_application(tmp_path / "workspace", background_workers=1) as app:
        project = app.projects.create(ProjectCreate("Background task"))

        def index_inputs(context, request):
            context.progress(0.25, "Scanning files")
            context.progress(0.75, "Building index")
            return {"indexed": request["count"]}

        app.background.register("index-inputs", index_inputs, in_process_safe=True)
        with pytest.raises(ValueError, match="non-simulation"):
            app.background.register(
                "simulation-run", index_inputs, in_process_safe=True,
            )
        task = app.background.submit(
            project.project_id, "index-inputs", {"count": 4},
        )
        completed = wait_for_task(app, task.task_id)
        assert completed.status is TaskStatus.COMPLETED
        assert completed.progress == 1
        assert completed.result == {"indexed": 4}
        notification = app.notifications.list(project_id=project.project_id)[0]
        assert notification.severity.value == "success"
        assert not notification.read
        assert app.notifications.mark_read(notification.notification_id).read


def test_database_rejects_newer_unknown_schema(tmp_path):
    path = tmp_path / "future.sqlite"
    database = AppDatabase(path)
    with database.transaction() as connection:
        connection.execute("PRAGMA user_version=99")
    database.close()
    with pytest.raises(RuntimeError, match="newer than supported"):
        AppDatabase(path)


def test_workspace_selection_and_layout_state_persist_across_restart(tmp_path):
    root = tmp_path / "workspace"
    with create_application(root, background_workers=1) as app:
        project = app.projects.create(ProjectCreate("Workspace state"))
        state = WorkspaceState(
            active_project_id=project.project_id,
            route=f"/projects/{project.project_id}/model/grid",
            selection={"cells": [1, 7], "field": "PORO"},
            bottom_panel="problems", bottom_panel_open=True, expert_mode=True,
        )
        app.state.save_workspace(state)
    with create_application(root, background_workers=1) as reopened:
        assert reopened.state.workspace() == state
        with pytest.raises(ValueError, match="unknown project"):
            reopened.state.save_workspace(WorkspaceState(
                active_project_id="project_missing", route="/projects/missing",
            ))
