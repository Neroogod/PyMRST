"""Acceptance coverage for App M2 model, patch, and grid inspection."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.app import create_application
from pymrst.app.database import APP_DATABASE_SCHEMA_VERSION
from pymrst.app.schemas import PatchOperation, ProjectCreate


def _project_with_deck(tmp_path, text: str):
    source = tmp_path / "CASE.DATA"
    source.write_text(text)
    app = create_application(tmp_path / "workspace", background_workers=1)
    project = app.projects.create(ProjectCreate("Inspection case"))
    asset = app.projects.import_input(project.project_id, source, kind="deck")
    return app, project, asset


def test_deck_inspection_indexes_sections_values_defaults_units_and_source(tmp_path):
    app, project, asset = _project_with_deck(tmp_path, """RUNSPEC
METRIC
DIMENS
  3 2 1 /
GRID
PORO
  3*0.20 2* 0.15 /
PROPS
SOLUTION
SCHEDULE
""")
    with app:
        model = app.models.inspect(project.project_id, asset.asset_id)
        assert model.unit_system == "METRIC"
        assert model.status == "ready"
        assert model.sections == ("RUNSPEC", "GRID", "PROPS", "SOLUTION", "SCHEDULE")
        poro = next(item for item in model.occurrences if item.keyword == "PORO")
        assert poro.value_count == 6
        assert poro.defaulted_value_count == 2
        source = app.models.source_lines(
            model.model_id, model.files[0]["path"], start_line=3, line_count=3,
        )
        assert source["lines"] == ["DIMENS", "  3 2 1 /", "GRID"]
        assert app.models.get(model.model_id) == model
        assert app.models.list(project.project_id) == (model,)


def test_deck_validation_collects_structure_unit_dimens_and_include_problems(tmp_path):
    app, project, asset = _project_with_deck(tmp_path, """GRID
FIELD
METRIC
DIMENS
  0 2 /
INCLUDE
  '../outside.inc' /
RUNSPEC
""")
    with app:
        model = app.models.inspect(project.project_id, asset.asset_id)
        codes = {item.code for item in model.issues}
        assert model.status == "invalid"
        assert model.unit_system == "UNKNOWN"
        assert {
            "deck.section_order", "units.conflict", "grid.invalid_dimens",
            "include.path_escape", "deck.section_missing",
        } <= codes
        assert all(item.location is not None for item in model.issues)


def test_missing_include_is_evidence_not_an_inspection_crash(tmp_path):
    app, project, asset = _project_with_deck(tmp_path, """RUNSPEC
METRIC
INCLUDE
  'MISSING.INC' /
GRID
PROPS
SOLUTION
SCHEDULE
""")
    with app:
        model = app.models.inspect(project.project_id, asset.asset_id)
        issue = next(item for item in model.issues if item.code == "include.missing")
        assert issue.severity.value == "error"
        assert "MISSING.INC" in issue.message


def test_patch_preview_and_materialization_never_change_original(tmp_path):
    app, project, asset = _project_with_deck(tmp_path, """RUNSPEC
METRIC
DIMENS
  2 1 1 /
GRID
PROPS
SOLUTION
SCHEDULE
""")
    with app:
        original_path = app.projects.asset_path(project.project_id, asset.asset_id)
        original = original_path.read_bytes()
        model = app.models.inspect(project.project_id, asset.asset_id)
        patch = app.patches.create(model.model_id, "Three cells", (
            PatchOperation("replace_lines", 4, 4, "  3 1 1 /"),
            PatchOperation("insert_before", 6, 6, "PORO\n  3*0.20 /"),
        ), description="Qualification variant")
        preview = app.patches.preview(patch.patch_id)
        assert "-  2 1 1 /" in preview["diff"]
        assert "+  3 1 1 /" in preview["diff"]
        assert preview["original_unchanged"]
        result = app.patches.materialize(patch.patch_id)
        variant = Path(project.root_path) / result.variant_relative_path
        assert result.status == "materialized"
        assert hashlib.sha256(variant.read_bytes()).hexdigest() == result.variant_hash
        assert original_path.read_bytes() == original
        lineage = json.loads((variant.parent / "lineage.json").read_text())
        assert lineage["base_hash"] == asset.content_hash
        assert lineage["patch_hash"] == patch.patch_hash
        assert app.patches.materialize(patch.patch_id) == result


def test_patch_rejects_overlaps_and_out_of_range_lines(tmp_path):
    app, project, asset = _project_with_deck(tmp_path, "RUNSPEC\nMETRIC\n")
    with app:
        model = app.models.inspect(project.project_id, asset.asset_id)
        with pytest.raises(ValueError, match="overlap"):
            app.patches.create(model.model_id, "bad", (
                PatchOperation("replace_lines", 1, 2, "RUNSPEC"),
                PatchOperation("delete_lines", 2, 2),
            ))
        with pytest.raises(ValueError, match="beyond"):
            app.patches.create(model.model_id, "bad", (
                PatchOperation("replace_lines", 9, 9, "X"),
            ))


def test_grid_snapshots_are_disk_backed_restart_safe_and_bounded(tmp_path):
    root = tmp_path / "workspace"
    app = create_application(root, background_workers=1)
    project = app.projects.create(ProjectCreate("Grid case"))
    grid = SimpleNamespace(
        cells=SimpleNamespace(
            num=4,
            centroids=np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=float),
            volumes=np.array([1, 1, 0.5, 0.5]),
            index_map=np.array([0, 1, 3, 4]),
        ),
        faces=SimpleNamespace(num=12),
        nodes=SimpleNamespace(num=9),
    )
    snapshot = app.grids.register(
        project.project_id, grid, name="Fault block",
        properties={"poro": [0.1, 0.2, np.nan, 0.3], "PERMX": [10, 20, 30, 40]},
    )
    assert snapshot.bounds == ((0.0, 1.0), (0.0, 1.0))
    assert next(item for item in snapshot.arrays if item.name == "PORO").nonfinite_count == 1
    app.close()

    with create_application(root, background_workers=1) as reopened:
        restored = reopened.grids.get(snapshot.snapshot_id)
        assert restored == snapshot
        values = reopened.grids.values(snapshot.snapshot_id, "poro", offset=1, count=2)
        assert values["shape"] == [2]
        assert values["values"][0] == 0.2
        assert np.isnan(values["values"][1])
        sampled = reopened.grids.values(
            snapshot.snapshot_id, "PERMX", count=2, stride=2,
        )
        assert sampled["values"] == [10, 30]
        with pytest.raises(ValueError, match="bounds"):
            reopened.grids.values(snapshot.snapshot_id, "poro", count=100_001)


def test_database_migrates_to_app_m2_schema(tmp_path):
    with create_application(tmp_path / "workspace", background_workers=1) as app:
        assert app.database.schema_version == APP_DATABASE_SCHEMA_VERSION
        names = {row[0] for row in app.database.fetch_all(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert {"app_models", "app_input_patches", "app_grid_snapshots"} <= names
