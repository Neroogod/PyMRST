"""M3.8 runnable supported subset and provenance refusal tests."""

import json

import numpy as np
import pytest

from pymrst.research_studio.eclipse import (
    RunnableWaterDeck, UnsupportedForExport, write_runnable_water_deck,
)
from pymrst.research_studio.evidence import capture_manifest


def _case():
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.params.rock import Rock
    grid = compute_geometry(cart_grid([2, 1, 1], [20.0, 10.0, 5.0]))
    rock = Rock(perm=np.array([[100.0, 50.0, 10.0], [120.0, 60.0, 12.0]]),
                poro=np.array([0.2, 0.25]))
    coord = []
    for y in (0.0, 10.0):
        for x in (0.0, 10.0, 20.0):
            coord.extend((x, y, 0.0, x, y, 5.0))
    grdecl = {"cartDims": [2, 1, 1], "COORD": coord,
              "ZCORN": [0.0]*8+[5.0]*8, "ACTNUM": [1, 1]}
    return grid, rock, grdecl


def test_water_deck_contains_all_sections_units_schedule_and_manifest(tmp_path):
    grid, rock, grdecl = _case()
    manifest = capture_manifest("run_m3_8", code_fingerprint="abc",
                                command=("pymrst", "run", "case.json"))
    specification = RunnableWaterDeck(
        "METRIC", 2.5e7, (1.0, 2.0), (2.5e7, 1.0, 4e-10, 1e-3, 0.0))
    path = write_runnable_water_deck(
        tmp_path / "WATER.DATA", grid, rock, grdecl=grdecl,
        specification=specification, manifest=manifest)
    text = path.read_text()
    for section in ("RUNSPEC", "GRID", "PROPS", "SOLUTION", "SUMMARY",
                    "SCHEDULE", "END"):
        assert section in text
    for keyword in ("METRIC", "WATER", "COORD", "ZCORN", "ACTNUM", "PORO",
                    "PERMX", "PERMY", "PERMZ", "PVTW", "PRESSURE", "TSTEP"):
        assert keyword in text
    recorded = json.loads(path.with_suffix(".pymrst-manifest.json").read_text())
    assert recorded["run_id"] == "run_m3_8"


def test_unsupported_physics_are_hard_failures(tmp_path):
    grid, rock, grdecl = _case()
    manifest = capture_manifest("run_m3_8", code_fingerprint="abc")
    specification = RunnableWaterDeck(
        "FIELD", 3000.0, (1.0,), (3000.0, 1.0, 1e-6, 1.0, 0.0))
    with pytest.raises(UnsupportedForExport, match="cannot represent"):
        write_runnable_water_deck(
            tmp_path / "BAD.DATA", grid, rock, grdecl=grdecl,
            specification=specification, manifest=manifest,
            unsupported_features=("wells", "relative permeability"))
    assert not (tmp_path / "BAD.DATA").exists()


def test_units_and_active_cell_mismatch_are_refused(tmp_path):
    grid, rock, grdecl = _case()
    with pytest.raises(UnsupportedForExport, match="METRIC or FIELD"):
        RunnableWaterDeck("SI", 1.0, (1.0,), (1, 1, 1, 1, 1))
    grdecl["ACTNUM"] = [1, 0]
    with pytest.raises(UnsupportedForExport, match="active count"):
        write_runnable_water_deck(
            tmp_path / "BAD.DATA", grid, rock, grdecl=grdecl,
            specification=RunnableWaterDeck(
                "METRIC", 1e7, (1,), (1e7, 1, 1e-9, 1e-3, 0)),
            manifest=capture_manifest("run", code_fingerprint="abc"))
