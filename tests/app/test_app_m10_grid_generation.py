"""Acceptance tests for generating a Cartesian grid without an existing deck."""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from pymrst.app import create_application
from pymrst.app.api.application import create_fastapi_app
from pymrst.app.schemas import ProjectCreate
from pymrst.app.schemas.deployment import ServerDeploymentSettings


def test_generate_cartesian_grid_registers_a_usable_snapshot(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Grid generation"))
        snapshot = app.grids.generate_cartesian(
            project.project_id, "Synthetic grid", [10, 8, 3], [1000.0, 800.0, 30.0],
        )
        assert snapshot.cell_count == 240
        assert snapshot.dimension == 3
        assert snapshot.name == "Synthetic grid"
        assert app.grids.get(snapshot.snapshot_id).snapshot_id == snapshot.snapshot_id
        assert [item.snapshot_id for item in app.grids.list(project.project_id)] == [snapshot.snapshot_id]


def test_generate_cartesian_grid_rejects_invalid_input(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Grid generation"))
        with pytest.raises(ValueError):
            app.grids.generate_cartesian(project.project_id, "bad", [0, 3, 2], [1.0, 1.0, 1.0])
        with pytest.raises(ValueError):
            app.grids.generate_cartesian(project.project_id, "bad", [3, 3, 2], [0.0, 1.0, 1.0])
        with pytest.raises(ValueError):
            app.grids.generate_cartesian(project.project_id, "huge", [200, 200, 200], [1.0, 1.0, 1.0])
        with pytest.raises(KeyError):
            app.grids.generate_cartesian("project_does-not-exist", "x", [1, 1, 1], [1.0, 1.0, 1.0])


@pytest.fixture
def http_application(tmp_path):
    context = create_application(tmp_path / "workspace")
    application = create_fastapi_app(context, ServerDeploymentSettings())
    with TestClient(application) as client:
        yield context, client
    context.close()


def test_generate_grid_endpoint_returns_a_grid_snapshot(http_application):
    _, client = http_application
    project = client.post("/api/v1/projects", json={
        "name": "HTTP grid generation", "description": "", "tags": [],
    }).json()
    response = client.post(
        f"/api/v1/projects/{project['project_id']}/grids/generate",
        json={"name": "Case grid", "dims": [6, 5, 2], "physical_dims_m": [600.0, 500.0, 20.0]},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["cell_count"] == 60
    assert body["name"] == "Case grid"
    listed = client.get(f"/api/v1/projects/{project['project_id']}/grids").json()
    assert [item["snapshot_id"] for item in listed] == [body["snapshot_id"]]


def test_generate_grid_endpoint_rejects_bad_dims_and_missing_project(http_application):
    _, client = http_application
    project = client.post("/api/v1/projects", json={
        "name": "HTTP grid generation", "description": "", "tags": [],
    }).json()
    missing_fields = client.post(
        f"/api/v1/projects/{project['project_id']}/grids/generate", json={"name": "x"},
    )
    assert missing_fields.status_code == 400
    bad_dims = client.post(
        f"/api/v1/projects/{project['project_id']}/grids/generate",
        json={"name": "x", "dims": [0, 1, 1], "physical_dims_m": [1.0, 1.0, 1.0]},
    )
    assert bad_dims.status_code == 400
    unknown_project = client.post(
        "/api/v1/projects/project_does-not-exist/grids/generate",
        json={"name": "x", "dims": [1, 1, 1], "physical_dims_m": [1.0, 1.0, 1.0]},
    )
    assert unknown_project.status_code == 404
