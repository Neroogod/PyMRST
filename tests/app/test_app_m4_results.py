"""Acceptance tests for App M4 state browsing, dashboards, comparison and export."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pymrst.app import create_application
from pymrst.app.database import APP_DATABASE_SCHEMA_VERSION
from pymrst.app.schemas import ProjectCreate


def create_result_run(app, project_id, *, pressure_offset=0.0, steps=3):
    configuration = app.runs.create_configuration(
        project_id, f"Synthetic {pressure_offset}", "protocol-qualification",
        {"steps": steps, "pressure_offset": pressure_offset},
    )
    run = app.runs.create_run(configuration.configuration_id)
    project = app.projects.get(project_id)
    root = Path(project.root_path) / run.run_relative_path
    state_root = root / "states"; state_root.mkdir()
    for step in range(1, steps + 1):
        relative = f"states/report_{step:04d}.npz"
        pressure = np.arange(6, dtype=float) + 100 * step + pressure_offset
        np.savez_compressed(
            root / relative,
            pressure=pressure,
            sW=np.linspace(0.1, 0.6, 6) + 0.01 * step,
            bhp=np.array([250.0-step, 310.0+step+pressure_offset]),
            qOs=np.array([-10.0-step, 0.0]),
            qWs=np.array([-1.0, 12.0+step]),
            qGs=np.array([-100.0, 0.0]),
        )
        app.runs.append_event(run.run_id, "state_saved", step, {
            "report_step": step, "relative_path": relative,
        })
        app.runs.append_event(run.run_id, "progress", step, {
            "progress": step / steps, "message": f"step {step}",
            "report_step": step, "report_steps": steps,
            "simulated_time_seconds": step * 86400.0,
        })
    return run


def test_state_index_is_checksum_bound_and_supports_bounded_field_slices(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Results"))
        run = create_result_run(app, project.project_id)
        states = app.results.index_run(run.run_id)
        assert [item.report_step for item in states] == [1, 2, 3]
        assert [item.simulated_time_seconds for item in states] == [86400, 172800, 259200]
        pressure = next(item for item in states[0].fields if item.name == "pressure")
        assert pressure.shape == (6,)
        assert pressure.minimum == 100
        values = app.results.field_values(states[1].state_id, "pressure", offset=2, count=3)
        assert values["values"] == [202.0, 203.0, 204.0]
        assert values["total"] == 6
        sampled = app.results.field_values(
            states[1].state_id, "pressure", offset=0, count=3, stride=2,
        )
        assert sampled["values"] == [200.0, 202.0, 204.0]
        assert sampled["stride"] == 2
        with pytest.raises(ValueError, match="bounds"):
            app.results.field_values(states[0].state_id, "pressure", count=250_001)
        assert app.results.states(run.run_id) == states


def test_field_and_well_dashboard_retains_full_time_history(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Dashboard"))
        run = create_result_run(app, project.project_id)
        dashboard = app.results.dashboard(run.run_id)
        assert dashboard["report_steps"] == [1, 2, 3]
        assert dashboard["times"] == [86400, 172800, 259200]
        assert dashboard["field"]["pressure"][0] == {
            "minimum": 100.0, "mean": 102.5, "maximum": 105.0,
        }
        assert dashboard["wells"]["bhp"][2] == [247.0, 313.0]
        assert dashboard["well_names"] == ["WELL_0", "WELL_1"]


def test_run_comparison_and_on_demand_difference_maps_are_aligned(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Comparison"))
        reference = create_result_run(app, project.project_id, pressure_offset=0)
        candidate = create_result_run(app, project.project_id, pressure_offset=2)
        comparison = app.results.compare(reference.run_id, candidate.run_id)
        pressure = comparison.summary["fields"]["pressure"]
        assert comparison.summary["common_report_steps"] == [1, 2, 3]
        assert pressure["maximum_absolute_error"] == 2.0
        assert pressure["root_mean_square_error"] == 2.0
        difference = app.results.difference(
            reference.run_id, candidate.run_id, 2, "pressure", offset=1, count=3,
        )
        assert difference["values"] == [2.0, 2.0, 2.0]
        assert difference["shape"] == [3]


def test_animation_manifest_and_csv_export_are_reproducible(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Export"))
        run = create_result_run(app, project.project_id)
        states = app.results.index_run(run.run_id)
        animation = app.results.animation_manifest(run.run_id, "sW")
        assert len(animation["frames"]) == 3
        assert animation["frames"][1]["simulated_time_seconds"] == 172800
        assert animation["frames"][1]["field_url"].endswith("/fields/sW")
        exported = app.results.export_field(states[0].state_id, "pressure", kind="csv")
        path = Path(project.root_path) / run.run_relative_path / exported.relative_path
        assert path.is_file()
        with path.open() as stream:
            rows = list(csv.reader(stream))
        assert rows[0] == ["index", "component", "pressure"]
        assert rows[1] == ["0", "0", "100.0"]
        assert len(rows) == 7


def test_result_tampering_is_detected_before_values_are_returned(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        project = app.projects.create(ProjectCreate("Tamper"))
        run = create_result_run(app, project.project_id, steps=1)
        state = app.results.index_run(run.run_id)[0]
        path = Path(project.root_path) / run.run_relative_path / state.relative_path
        path.write_bytes(b"tampered")
        with pytest.raises(RuntimeError, match="lineage"):
            app.results.field_values(state.state_id, "pressure")


def test_app_m4_schema_tables_exist(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        assert app.database.schema_version == APP_DATABASE_SCHEMA_VERSION
        names = {row[0] for row in app.database.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert {"app_result_states", "app_result_comparisons", "app_result_exports"} <= names
