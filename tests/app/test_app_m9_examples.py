"""Acceptance tests for the curated, run-in-app PyMRST example catalog."""

from __future__ import annotations

import time

import pytest

from pymrst.app import create_application
from pymrst.app.schemas import ProjectCreate, RunStatus, TaskStatus


def wait_for_run(app, run_id, timeout=60):
    terminal = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = app.runs.get(run_id)
        if run.status in terminal:
            return run
        time.sleep(0.05)
    raise TimeoutError(run_id)


def test_catalog_lists_curated_self_contained_examples(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        items = app.examples.list()
        assert len(items) >= 10
        ids = [item.example_id for item in items]
        assert len(ids) == len(set(ids)), "example ids must be unique"
        assert "buckley-leverett-1d" in ids
        for item in items:
            assert item.title and item.description and item.category
            assert item.runtime_class.value in {"fast", "medium", "slow"}
        assert app.examples.get("buckley-leverett-1d").title == "Buckley-Leverett displacement"
        with pytest.raises(KeyError):
            app.examples.get("does-not-exist")


@pytest.mark.parametrize("example_id", [
    "buckley-leverett-1d", "mandel-poroelasticity", "grid-factory-tour",
])
def test_example_runs_to_completion_in_an_isolated_process(tmp_path, example_id):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Examples"))
        configuration = app.runs.create_configuration(
            project.project_id, "Example run", "pymrst-example", {"example_id": example_id},
        )
        run = wait_for_run(app, app.isolated.submit(configuration.configuration_id).run_id)
        task = app.tasks.get(run.task_id)
        assert run.status is RunStatus.COMPLETED, task.error
        assert task.status is TaskStatus.COMPLETED
        assert task.result["example_id"] == example_id
        assert task.result["title"]
        assert task.result["result"] is not None
        progress_events = [item for item in app.runs.events(run.run_id) if item.event_type == "progress"]
        assert progress_events, "example run should report at least one progress event"


def test_unknown_example_id_fails_the_run_cleanly(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Examples"))
        configuration = app.runs.create_configuration(
            project.project_id, "Bad example", "pymrst-example", {"example_id": "not-a-real-example"},
        )
        run = wait_for_run(app, app.isolated.submit(configuration.configuration_id).run_id)
        task = app.tasks.get(run.task_id)
        assert run.status is RunStatus.FAILED
        assert task.status is TaskStatus.FAILED
        assert task.error["error_type"] == "KeyError"
        assert "not-a-real-example" in task.error["message"]
