"""Acceptance coverage for App M5 governed model construction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pymrst.app import create_application
from pymrst.app.database import APP_DATABASE_SCHEMA_VERSION
from pymrst.app.schemas import (
    ControlPeriod, FieldSource, FluidDefinition, InitialStateDefinition,
    ModelConstructionDocument, ProjectCreate, RockDefinition,
    ScheduleDefinition, WellDefinition,
)
from pymrst.gridprocessing import cart_grid, compute_geometry


def _document(*, porosity=0.22, producer_cell=5):
    return ModelConstructionDocument(
        rock=RockDefinition(
            porosity=FieldSource("grid_array", array="PORO"),
            permeability_x=FieldSource("grid_array", array="PERMX"),
            permeability_y=FieldSource("uniform", value=80e-15),
            permeability_z=FieldSource("uniform", value=10e-15),
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
                "W": FieldSource("uniform", value=porosity),
                "O": FieldSource("uniform", value=1-porosity),
            },
        ),
        wells=(
            WellDefinition("INJ", (0,), "rate", 1e-4, composition=(1.0, 0.0), sign=1),
            WellDefinition("PROD", (producer_cell,), "bhp", 15e6,
                           composition=(0.0, 1.0), sign=-1),
        ),
        schedule=ScheduleDefinition(
            timestep_seconds=(86400.0, 172800.0, 172800.0),
            step_control=(0, 1, 1),
            controls=(
                ControlPeriod("startup"),
                ControlPeriod("production", {
                    "INJ": {"target_si": 2e-4},
                    "PROD": {"target_si": 14e6, "limits": {"orat": -2e-4}},
                }),
            ),
        ),
    )


def _case(tmp_path):
    app = create_application(tmp_path / "workspace")
    project = app.projects.create(ProjectCreate("Constructed waterflood"))
    grid = compute_geometry(cart_grid([3, 2, 1], [300.0, 200.0, 10.0]))
    snapshot = app.grids.register(
        project.project_id, grid, name="Six-cell grid",
        properties={"PORO": np.full(6, 0.22), "PERMX": np.full(6, 100e-15)},
    )
    return app, project, grid, snapshot


def test_draft_is_validated_versioned_and_audited(tmp_path):
    app, project, _, snapshot = _case(tmp_path)
    with app:
        draft = app.construction.create(
            project.project_id, snapshot.snapshot_id, "Base waterflood", _document(),
            actor="reservoir-engineer",
        )
        assert draft.status == "ready"
        assert draft.version == 1
        assert not draft.issues
        history = app.construction.history(draft.draft_id)
        assert history[0]["actor"] == "reservoir-engineer"
        assert history[0]["document_hash"] == draft.document_hash

        changed = app.construction.update(
            draft.draft_id, _document(porosity=0.25), expected_version=1,
            reason="raise initial water saturation",
        )
        assert changed.version == 2
        assert changed.document_hash != draft.document_hash
        assert len(app.construction.history(draft.draft_id)) == 2
        with pytest.raises(RuntimeError, match="version conflict"):
            app.construction.update(
                draft.draft_id, _document(), expected_version=1,
            )


def test_engineering_errors_block_materialization(tmp_path):
    app, project, _, snapshot = _case(tmp_path)
    with app:
        invalid = _document(porosity=1.2, producer_cell=99)
        draft = app.construction.create(
            project.project_id, snapshot.snapshot_id, "Invalid", invalid,
        )
        codes = {item["code"] for item in draft.issues}
        assert draft.status == "invalid"
        assert {"initialization.saturation_range", "wells.invalid_completion"} <= codes
        with pytest.raises(ValueError, match="cannot be materialized"):
            app.construction.materialize(draft.draft_id)


def test_materialized_revision_is_immutable_deduplicated_and_compilable(tmp_path):
    app, project, grid, snapshot = _case(tmp_path)
    with app:
        draft = app.construction.create(
            project.project_id, snapshot.snapshot_id, "Base waterflood", _document(),
        )
        revision = app.construction.materialize(draft.draft_id)
        assert app.construction.materialize(draft.draft_id) == revision
        assert app.construction.revisions(draft.draft_id) == (revision,)
        path = Path(project.root_path) / revision.relative_path
        payload = json.loads(path.read_text())
        assert payload["content_hash"] == revision.content_hash
        assert not bool(path.stat().st_mode & 0o200)

        compiled = app.construction.compile(revision.revision_id, grid)
        assert compiled.rock.perm.shape == (6, 3)
        assert np.allclose(compiled.initial_state["s"].sum(axis=1), 1.0)
        assert [well.name for well in compiled.wells] == ["INJ", "PROD"]
        assert compiled.schedule.n_steps == 3
        assert compiled.schedule.total_time == 5*86400
        assert compiled.schedule.controls[1].W[0].val == 2e-4
        assert compiled.schedule.controls[1].W[1].val == 14e6
        assert compiled.schedule.controls[1].W[1].lims == {"orat": -2e-4}

        app.construction.update(
            draft.draft_id, _document(porosity=0.3), expected_version=1,
        )
        original = app.construction.compile(revision.revision_id, grid)
        assert np.allclose(original.initial_state["sW"], 0.22)


def test_revision_tampering_is_detected_before_compilation(tmp_path):
    app, project, grid, snapshot = _case(tmp_path)
    with app:
        draft = app.construction.create(
            project.project_id, snapshot.snapshot_id, "Base waterflood", _document(),
        )
        revision = app.construction.materialize(draft.draft_id)
        path = Path(project.root_path) / revision.relative_path
        path.chmod(0o644)
        payload = json.loads(path.read_text())
        payload["document"]["fluid"]["density_kg_m3"][0] = 999.0
        path.write_text(json.dumps(payload))
        with pytest.raises(RuntimeError, match="checksum"):
            app.construction.compile(revision.revision_id, grid)


def test_grid_property_source_and_dimension_basis_are_checked(tmp_path):
    app, project, _, snapshot = _case(tmp_path)
    with app:
        missing = _document()
        missing = ModelConstructionDocument(
            RockDefinition(
                FieldSource("grid_array", array="MISSING"),
                FieldSource("uniform", value=1e-13),
                FieldSource("uniform", value=1e-13),
            ),
            missing.fluid, missing.initial_state, missing.wells, missing.schedule,
        )
        issues = app.construction.validate(snapshot.snapshot_id, missing)
        assert {item["code"] for item in issues} >= {
            "field.invalid_source", "rock.permeability_basis",
        }


def test_app_m5_schema_tables_exist(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        assert app.database.schema_version == APP_DATABASE_SCHEMA_VERSION
        names = {row[0] for row in app.database.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert {"app_model_drafts", "app_model_draft_events", "app_model_revisions"} <= names
