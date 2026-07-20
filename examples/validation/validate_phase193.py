"""Phase 193: MATLAB-struct-like compositional facility deck conversion."""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.gridprocessing import cart_grid, compute_geometry  # noqa: E402
from pymrst.modules.compositional.facility import (  # noqa: E402
    apply_deck_well_controls,
    facility_group_controls_from_deck_control,
    facility_group_controls_from_deck_schedule,
    wells_from_deck_completion_records,
)
from pymrst.params.rock import make_rock  # noqa: E402
from pymrst.params.wells_and_bc import add_well  # noqa: E402
from pymrst.utils.units import BARSA, DARCY, MILLI  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _grid_rock():
    grid = compute_geometry(cart_grid([2, 1, 2], [20.0, 10.0, 20.0]))
    rock = make_rock(grid, 100.0 * MILLI * DARCY, 0.25)
    return grid, rock


def _object_completion_control():
    return SimpleNamespace(
        WELSPECS=np.array([
            SimpleNamespace(well="P1", group="PROD", i=1, j=1, ref_depth=1000.0),
            SimpleNamespace(well="INJ", group="INJGRP", i=2, j=1, ref_depth=1000.0),
        ], dtype=object),
        COMPDAT=np.array([
            SimpleNamespace(well="P1", i=1, j=1, k1=1, k2=2, status="OPEN",
                            diameter=0.2, kh=50.0, skin=0.0, dir="Z"),
            SimpleNamespace(well="INJ", i=2, j=1, k1=1, k2=1, status="OPEN",
                            wi=1.0e-12, radius=0.05, dir="Z"),
        ], dtype=object),
    )


def validate_struct_like_completion_import() -> int:
    grid, rock = _grid_rock()
    wells = wells_from_deck_completion_records(
        grid,
        rock,
        _object_completion_control(),
        well_type="bhp",
        val=75.0 * BARSA,
        compi=[0.0, 1.0],
    )

    failures = 0
    failures += _check("object WELSPECS/COMPDAT import wells",
                       len(wells) == 2
                       and wells[0].name == "P1"
                       and wells[1].name == "INJ")
    failures += _check("object COMPDAT expands K range and preserves metadata",
                       len(wells[0].cells) == 2
                       and getattr(wells[0], "group") == "PROD"
                       and np.all(np.asarray(getattr(wells[0], "kh")) > 0.0)
                       and np.all(np.asarray(getattr(wells[0], "cstatus"))))
    failures += _check("object COMPDAT explicit WI survives",
                       len(wells[1].cells) == 1
                       and np.isclose(wells[1].WI[0], 1.0e-12))
    return failures


def validate_struct_like_well_controls() -> int:
    grid, rock = _grid_rock()
    wells = add_well(None, grid, rock, np.array([0]), well_type="bhp",
                     val=80.0 * BARSA, compi=np.array([0.0, 1.0]),
                     sign=-1, name="P1")
    wells = add_well(wells, grid, rock, np.array([1]), well_type="rate",
                     val=1.0e-5, compi=np.array([0.0, 1.0]),
                     sign=1, name="INJ")
    control = SimpleNamespace(
        WCONPROD=np.array([
            SimpleNamespace(well="P1", status="OPEN", control="ORAT",
                            orat=2.0e-5, bhp=70.0 * BARSA, vfp_index=3)
        ], dtype=object),
        WCONINJE=np.array([
            SimpleNamespace(well="INJ", phase="GAS", status="OPEN",
                            control="RATE", rate=1.5e-5, bhp=120.0 * BARSA)
        ], dtype=object),
        WELTARG=np.array([
            SimpleNamespace(well="P1", mode="BHP", value=65.0 * BARSA)
        ], dtype=object),
    )
    updated = apply_deck_well_controls(wells, control, grid=grid)

    failures = 0
    failures += _check("object WCONPROD applies producer control",
                       updated[0].sign == -1
                       and updated[0].type == "orat"
                       and np.isclose(updated[0].val, -2.0e-5)
                       and np.isclose(updated[0].lims["bhp"], 65.0 * BARSA)
                       and updated[0].vfp_index == 3)
    failures += _check("object WCONINJE applies injector control",
                       updated[1].sign == 1
                       and updated[1].type == "rate"
                       and np.isclose(updated[1].val, 1.5e-5)
                       and np.allclose(updated[1].compi, [0.0, 1.0]))
    failures += _check("object controls do not mutate base wells",
                       wells[0].type == "bhp"
                       and wells[1].val == 1.0e-5)
    return failures


def validate_struct_like_group_controls() -> int:
    control = SimpleNamespace(
        GRUPTREE=np.array([
            SimpleNamespace(child="WEST", parent="FIELD"),
        ], dtype=object),
        WELSPECS=np.array([
            SimpleNamespace(well="P1", group="WEST"),
            SimpleNamespace(well="P2", group="WEST"),
        ], dtype=object),
        WGRUPCON=np.array([
            SimpleNamespace(well="P1", status="YES", guide_rate=2.0,
                            group="WEST", guide_multiplier=3.0),
            SimpleNamespace(well="P2", status="YES", guide_rate=1.0,
                            group="WEST", guide_multiplier=1.0),
        ], dtype=object),
        GCONPROD=np.array([
            SimpleNamespace(group="FIELD", control="ORAT", orat=1.0e-5,
                            status="YES", exceed_action="NONE")
        ], dtype=object),
        GCONINJE=np.array([
            SimpleNamespace(group="INJGRP", phase="GAS", control="RATE",
                            surface=5.0e-5, status="YES")
        ], dtype=object),
        GECON=np.array([
            SimpleNamespace(group="FIELD", min_oil_rate=1.0e-6, end_run="YES")
        ], dtype=object),
        GRUPNET=np.array([
            SimpleNamespace(group="WEST", guide_rate=10.0, use_network="YES")
        ], dtype=object),
    )
    groups = facility_group_controls_from_deck_control(control)
    by_name = {group.name: group for group in groups}
    schedule_groups = facility_group_controls_from_deck_schedule(
        SimpleNamespace(control=np.array([control], dtype=object))
    )

    failures = 0
    failures += _check("object group control imports tree and limits",
                       set(by_name) >= {"WEST", "FIELD", "INJGRP"}
                       and "group:WEST" in by_name["FIELD"].wells
                       and np.isclose(by_name["FIELD"].lims["orat"], 1.0e-5)
                       and np.isclose(by_name["INJGRP"].lims["grat"], 5.0e-5))
    failures += _check("object WGRUPCON/GRUPNET metadata survives",
                       by_name["WEST"].allocation == "fraction"
                       and by_name["WEST"].fractions == [6.0, 1.0]
                       and by_name["WEST"].extra["GRUPNET"]["use_network"])
    failures += _check("object GCON/GECON raw metadata survives",
                       by_name["FIELD"].extra["GCONPROD"][0]["control"] == "orat"
                       and by_name["FIELD"].extra["GECON"]["end_run"])
    failures += _check("object schedule converts every facility control block",
                       len(schedule_groups) == 1
                       and any(group.name == "FIELD" for group in schedule_groups[0]))
    return failures


def main() -> int:
    print("\nPhase 193: MATLAB-struct-like compositional facility deck conversion")
    failures = validate_struct_like_completion_import()
    failures += validate_struct_like_well_controls()
    failures += validate_struct_like_group_controls()
    if failures:
        print(f"\nPhase 193 failed ({failures} checks).")
        return 1
    print("\nFacility deck converters now accept MATLAB-style struct records directly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
