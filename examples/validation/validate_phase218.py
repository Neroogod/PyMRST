"""Phase 218: case-insensitive generic/deck schedule metadata cleanup."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    schedule_controls_from_deck_controls,
    simple_schedule,
    simple_schedule_from_deck_controls,
)
from pymrst.modules.compositional.examples.compositional_boundary_conditions_example import (  # noqa: E402
    make_boundary_conditions_case,
)
from pymrst.modules.compositional.examples._shared import make_line_drive_case  # noqa: E402
from pymrst.params.rock import pore_volume  # noqa: E402
from pymrst.params.wells_and_bc import add_source  # noqa: E402
from pymrst.utils.units import DAY  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _native_source():
    grid, _, _, _, _, model, _ = make_line_drive_case(nx=4)
    rate = float(pore_volume(grid, model.rock).sum() / (1000.0 * DAY))
    src = add_source(
        None,
        np.asarray([0, grid.cells.num - 1], dtype=np.int64),
        np.asarray([rate, -rate], dtype=float),
        sat=np.asarray([[0.0, 1.0], [0.5, 0.5]], dtype=float),
        components=np.asarray([[0.2, 0.7, 0.1], [0.0, 0.0, 0.0]], dtype=float),
    )
    return src


def _side_blocks():
    _, _, _, _, wells, _, _ = make_line_drive_case(nx=4)
    return wells, {"name": "sep"}, [{"name": "FIELD"}]


def validate_simple_schedule_extra_cleanup() -> int:
    native_bc = make_boundary_conditions_case(nx=4, ny=2)["bc"]
    native_src = _native_source()
    wells, separator, groups = _side_blocks()
    schedule = simple_schedule(
        [1.0],
        control=[
            {
                "WELLS": wells,
                "BOUNDARYCONDITIONS": native_bc,
                "SOURCETERMS": native_src,
                "SEPARATORGROUP": separator,
                "GROUPCONTROLS": groups,
                "VFPPROD": [],
                "Label": "simple-clean",
            }
        ],
    )
    control = schedule.control[0]
    failures = 0
    failures += _check("simple_schedule consumes uppercase well alias",
                       control.wells is wells)
    failures += _check("simple_schedule consumes uppercase native forcing aliases",
                       control.bc is not None and control.src is not None)
    failures += _check("simple_schedule consumes uppercase facility aliases",
                       control.separator_group is separator and control.group_controls is groups)
    failures += _check("simple_schedule filters consumed uppercase aliases",
                       control.extra == {"Label": "simple-clean"})
    return failures


def validate_deck_schedule_extra_cleanup() -> int:
    native_bc = make_boundary_conditions_case(nx=4, ny=2)["bc"]
    native_src = _native_source()
    wells, separator, groups = _side_blocks()
    direct = schedule_controls_from_deck_controls(
        [],
        [
            {
                "WELLS": wells,
                "BCRECORDS": native_bc,
                "SRCRECORDS": native_src,
                "SEPARATORGROUP": separator,
                "GROUPCONTROLS": groups,
                "VFPINJ": [],
                "Tag": "deck-clean",
            }
        ],
    )[0]
    wrapped = simple_schedule_from_deck_controls(
        [1.0],
        [],
        [
            {
                "WELLS": wells,
                "BOUNDARYCONDITIONS": native_bc,
                "SOURCETERMS": native_src,
                "SEPARATORGROUP": separator,
                "GROUPCONTROLS": groups,
                "VFPPROD": [],
                "Tag": "wrapped-clean",
            }
        ],
    ).control[0]

    failures = 0
    for name, control, tag in (
        ("deck conversion", direct, "deck-clean"),
        ("deck simple schedule", wrapped, "wrapped-clean"),
    ):
        failures += _check(f"{name} consumes uppercase side aliases",
                           control.wells
                           and control.bc is not None
                           and control.src is not None
                           and control.separator_group is separator
                           and control.group_controls
                           and control.group_controls[0]["name"] == "FIELD")
        failures += _check(f"{name} filters consumed uppercase aliases",
                           control.extra == {"Tag": tag})
    return failures


def main() -> int:
    print("\nPhase 218: case-insensitive generic/deck schedule metadata cleanup")
    failures = validate_simple_schedule_extra_cleanup()
    failures += validate_deck_schedule_extra_cleanup()
    if failures:
        print(f"\nPhase 218 failed ({failures} checks).")
        return 1
    print("\nGeneric and deck schedules now filter consumed aliases case-insensitively.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
