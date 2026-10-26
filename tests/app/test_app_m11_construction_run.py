"""Acceptance tests for running a materialized model-construction revision.

Confirms the end-to-end path a reservoir engineer actually needs: generate a
grid with no deck file, define rock/fluid/wells/schedule in the guided
builder, materialize an immutable revision, and run it through the isolated
worker process — not just the fixed SPE1 benchmark.
"""

from __future__ import annotations

import time

import pytest

from pymrst.app import create_application
from pymrst.app.schemas import (
    ControlPeriod, FieldSource, FluidDefinition, InitialStateDefinition,
    ModelConstructionDocument, ProjectCreate, RockDefinition, RunStatus,
    ScheduleDefinition, TaskStatus, WellDefinition,
)

DAY = 86400.0


def wait_for_run(app, run_id, timeout=60):
    terminal = {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = app.runs.get(run_id)
        if run.status in terminal:
            return run
        time.sleep(0.05)
    raise TimeoutError(run_id)


def _document():
    return ModelConstructionDocument(
        rock=RockDefinition(
            porosity=FieldSource("uniform", value=0.2),
            permeability_x=FieldSource("uniform", value=120e-15),
            permeability_y=FieldSource("uniform", value=120e-15),
            permeability_z=FieldSource("uniform", value=15e-15),
        ),
        fluid=FluidDefinition(
            phases=("W", "O"), viscosity_pa_s=(1e-3, 5e-3),
            density_kg_m3=(1000.0, 800.0), relperm_exponent=(2.0, 2.0),
            compressibility_pa_inverse=(4e-10, 1e-9),
            reference_pressure_pa=20e6,
        ),
        initial_state=InitialStateDefinition(
            pressure=FieldSource("uniform", value=20e6),
            saturations={
                "W": FieldSource("uniform", value=0.2),
                "O": FieldSource("uniform", value=0.8),
            },
        ),
        wells=(
            WellDefinition("INJ", (0,), "rate", 1e-4, composition=(1.0, 0.0), sign=1),
            WellDefinition("PROD", (23,), "bhp", 15e6, composition=(0.0, 1.0), sign=-1),
        ),
        schedule=ScheduleDefinition(
            timestep_seconds=(2*DAY, 2*DAY, 2*DAY),
            step_control=(0, 0, 0),
            controls=(ControlPeriod("base"),),
        ),
    )


def _materialized_revision(app, project_id):
    snapshot = app.grids.generate_cartesian(project_id, "Case grid", [4, 3, 2], [400.0, 300.0, 20.0])
    draft = app.construction.create(project_id, snapshot.snapshot_id, "Waterflood", _document())
    assert draft.status == "ready", draft.issues
    return app.construction.materialize(draft.draft_id)


def test_run_completes_against_a_generated_grid_and_reports_evidence(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Constructed run"))
        revision = _materialized_revision(app, project.project_id)
        configuration = app.runs.create_configuration(
            project.project_id, "Run constructed model", "pymrst-construction-run",
            {"revision_id": revision.revision_id},
        )
        run = wait_for_run(app, app.isolated.submit(configuration.configuration_id).run_id)
        task = app.tasks.get(run.task_id)
        assert run.status is RunStatus.COMPLETED, task.error
        assert task.status is TaskStatus.COMPLETED
        assert task.result["revision_id"] == revision.revision_id
        assert task.result["report_steps"] == 3
        assert task.result["simulated_time_seconds"] == pytest.approx(6*DAY)
        assert task.result["final_pressure_min"] > 0
        progress_events = [item for item in app.runs.events(run.run_id) if item.event_type == "progress"]
        assert progress_events
        convergence_events = [item for item in app.runs.events(run.run_id) if item.event_type == "convergence"]
        assert len(convergence_events) == 3


def test_run_fails_cleanly_for_an_unknown_revision(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Constructed run"))
        configuration = app.runs.create_configuration(
            project.project_id, "Bad revision", "pymrst-construction-run",
            {"revision_id": "revision_does-not-exist"},
        )
        run = wait_for_run(app, app.isolated.submit(configuration.configuration_id).run_id)
        task = app.tasks.get(run.task_id)
        assert run.status is RunStatus.FAILED
        assert task.status is TaskStatus.FAILED
        assert task.error["error_type"] == "KeyError"


def test_project_revisions_lists_every_materialized_revision(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Constructed run"))
        assert app.construction.project_revisions(project.project_id) == ()
        revision = _materialized_revision(app, project.project_id)
        listed = app.construction.project_revisions(project.project_id)
        assert [item.revision_id for item in listed] == [revision.revision_id]
        assert listed[0].summary["name"] == "Waterflood"
        assert listed[0].summary["cells"] == 24
        assert listed[0].summary["wells"] == 2


@pytest.fixture
def http_application(tmp_path):
    from fastapi.testclient import TestClient
    from pymrst.app.api.application import create_fastapi_app
    from pymrst.app.schemas.deployment import ServerDeploymentSettings
    pytest.importorskip("fastapi"); pytest.importorskip("httpx")
    context = create_application(tmp_path / "workspace")
    application = create_fastapi_app(context, ServerDeploymentSettings())
    with TestClient(application) as client:
        yield context, client
    context.close()


def test_http_run_lifecycle_against_a_constructed_model(http_application):
    _, client = http_application
    project = client.post("/api/v1/projects", json={
        "name": "HTTP constructed run", "description": "", "tags": [],
    }).json()
    grid = client.post(f"/api/v1/projects/{project['project_id']}/grids/generate", json={
        "name": "Case grid", "dims": [4, 3, 2], "physical_dims_m": [400.0, 300.0, 20.0],
    }).json()
    document = _document().to_dict()
    draft = client.post(f"/api/v1/projects/{project['project_id']}/model-drafts", json={
        "name": "HTTP waterflood", "grid_snapshot_id": grid["snapshot_id"], "document": document,
    })
    assert draft.status_code == 201, draft.text
    revision = client.post(f"/api/v1/model-drafts/{draft.json()['draft_id']}/materialize")
    assert revision.status_code == 201, revision.text

    listed = client.get(f"/api/v1/projects/{project['project_id']}/model-revisions")
    assert listed.status_code == 200
    assert [item["revision_id"] for item in listed.json()] == [revision.json()["revision_id"]]

    configuration = client.post(f"/api/v1/projects/{project['project_id']}/run-configurations", json={
        "name": "Run it", "runner_kind": "pymrst-construction-run", "model_id": "",
        "payload": {"revision_id": revision.json()["revision_id"]},
    })
    assert configuration.status_code == 201, configuration.text
    run = client.post(f"/api/v1/run-configurations/{configuration.json()['configuration_id']}/submit")
    assert run.status_code == 202, run.text
    run_id = run.json()["run_id"]

    terminal = {"completed", "failed", "cancelled"}
    deadline = time.monotonic() + 60
    status = None
    while time.monotonic() < deadline:
        status = client.get(f"/api/v1/runs/{run_id}").json()["status"]
        if status in terminal:
            break
        time.sleep(0.05)
    assert status == "completed", client.get(f"/api/v1/runs/{run_id}/events").json()
