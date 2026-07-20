"""Phase 216: case-insensitive schedule side facility blocks."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    schedule_controls_from_boundary_controls,
    schedule_controls_from_forcing_controls,
    schedule_controls_from_source_controls,
    simple_schedule,
    simple_schedule_from_mrst_schedule,
)
from pymrst.modules.compositional.examples.compositional_boundary_conditions_example import (  # noqa: E402
    make_boundary_conditions_case,
)
from pymrst.modules.compositional.examples._shared import make_line_drive_case  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _side_blocks():
    _, _, _, _, wells, _, _ = make_line_drive_case(nx=4)
    separator = {"name": "sep-A"}
    groups = [{"name": "FIELD", "limit": "orat"}]
    return wells, separator, groups


def validate_simple_schedule_aliases() -> int:
    wells, separator, groups = _side_blocks()
    schedule = simple_schedule(
        [1.0, 2.0],
        control=[
            {
                "Wells": wells,
                "SeparatorGroup": separator,
                "GroupControls": groups,
                "Label": "mixed-side",
            }
        ],
    )
    control = schedule.control[0]
    failures = 0
    failures += _check("simple_schedule consumes Wells alias",
                       control.wells is wells)
    failures += _check("simple_schedule consumes SeparatorGroup alias",
                       control.separator_group is separator)
    failures += _check("simple_schedule consumes GroupControls alias",
                       control.group_controls is groups)
    failures += _check("consumed side aliases do not leak into extra",
                       control.extra == {"Label": "mixed-side"})
    return failures


def validate_mrst_schedule_aliases() -> int:
    wells, separator, groups = _side_blocks()
    raw = {
        "step": {"val": np.asarray([1.0]), "control": np.asarray([1])},
        "control": [
            {
                "WELLS": wells,
                "separatorGroup": separator,
                "GROUPCONTROLS": groups,
                "Tag": "mrst-side",
            }
        ],
    }
    schedule = simple_schedule_from_mrst_schedule(raw)
    control = schedule.control[0]
    failures = 0
    failures += _check("MRST-like schedule consumes WELLS alias",
                       control.wells is wells)
    failures += _check("MRST-like schedule consumes separatorGroup alias",
                       control.separator_group is separator)
    failures += _check("MRST-like schedule consumes GROUPCONTROLS alias",
                       control.group_controls is groups)
    failures += _check("MRST-like side aliases do not leak into extra",
                       control.extra == {"Tag": "mrst-side"})
    return failures


def validate_forcing_schedule_aliases() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    wells, separator, groups = _side_blocks()
    grid = case["grid"]

    boundary = schedule_controls_from_boundary_controls(
        grid,
        [
            {
                "WELLS": wells,
                "separatorGroup": separator,
                "GROUPCONTROLS": groups,
                "SIDE": "xmin",
                "TYPE": "flux",
                "Q": 0.0,
                "Label": "bc-side",
            }
        ],
    )[0]
    source = schedule_controls_from_source_controls(
        [
            {
                "Wells": wells,
                "SeparatorGroup": separator,
                "GroupControls": groups,
                "CELL": 0,
                "Q": 0.0,
                "Label": "src-side",
            }
        ],
    )[0]
    forcing = schedule_controls_from_forcing_controls(
        grid,
        [
            {
                "WELLS": wells,
                "SeparatorGroup": separator,
                "GROUPCONTROLS": groups,
                "SIDE": "xmin",
                "Q": 0.0,
                "CELL": 0,
                "RATE": 0.0,
                "Label": "forcing-side",
            }
        ],
    )[0]

    failures = 0
    for name, control, label in (
        ("boundary", boundary, "bc-side"),
        ("source", source, "src-side"),
        ("mixed forcing", forcing, "forcing-side"),
    ):
        failures += _check(f"{name} schedule consumes well side alias",
                           control.wells is wells)
        failures += _check(f"{name} schedule consumes separator side alias",
                           control.separator_group is separator)
        failures += _check(f"{name} schedule consumes group side alias",
                           control.group_controls is groups)
        failures += _check(f"{name} consumed side aliases do not leak",
                           control.extra == {"Label": label})
    return failures


def main() -> int:
    print("\nPhase 216: case-insensitive schedule side facility blocks")
    failures = validate_simple_schedule_aliases()
    failures += validate_mrst_schedule_aliases()
    failures += validate_forcing_schedule_aliases()
    if failures:
        print(f"\nPhase 216 failed ({failures} checks).")
        return 1
    print("\nSchedule well/separator/group side blocks now accept MRST-style capitalization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
