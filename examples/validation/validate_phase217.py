"""Phase 217: native forcing aliases in generic and deck schedules."""

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
    return src, rate


def validate_simple_schedule_native_aliases() -> int:
    native_bc = make_boundary_conditions_case(nx=4, ny=2)["bc"]
    native_src, rate = _native_source()
    schedule = simple_schedule(
        [1.0],
        control=[
            {
                "BoundaryConditions": native_bc,
                "SourceTerms": native_src,
                "Label": "simple-native",
            }
        ],
    )
    control = schedule.control[0]
    failures = 0
    failures += _check("simple_schedule consumes native BoundaryConditions alias",
                       control.bc is not None
                       and control.bc is not native_bc
                       and np.array_equal(control.bc.face, native_bc.face))
    failures += _check("simple_schedule consumes native SourceTerms alias",
                       control.src is not None
                       and control.src is not native_src
                       and np.allclose(control.src.rate, [rate, -rate]))
    failures += _check("simple_schedule native forcing aliases are consumed",
                       control.extra == {"Label": "simple-native"})
    return failures


def validate_deck_schedule_native_aliases() -> int:
    native_bc = make_boundary_conditions_case(nx=4, ny=2)["bc"]
    native_src, rate = _native_source()
    controls = schedule_controls_from_deck_controls(
        [],
        [
            {
                "BCRecords": native_bc,
                "SrcRecords": native_src,
                "Tag": "deck-native",
            }
        ],
    )
    schedule = simple_schedule_from_deck_controls(
        [1.0],
        [],
        [
            {
                "BoundaryConditions": native_bc,
                "SourceTerms": native_src,
                "Tag": "simple-deck-native",
            }
        ],
    )

    direct = controls[0]
    wrapped = schedule.control[0]
    failures = 0
    for name, control, tag in (
        ("deck conversion", direct, "deck-native"),
        ("deck simple schedule", wrapped, "simple-deck-native"),
    ):
        failures += _check(f"{name} consumes native BC alias",
                           control.bc is not None
                           and control.bc is not native_bc
                           and np.array_equal(control.bc.face, native_bc.face))
        failures += _check(f"{name} consumes native source alias",
                           control.src is not None
                           and control.src is not native_src
                           and np.allclose(control.src.rate, [rate, -rate]))
        failures += _check(f"{name} native forcing aliases are consumed",
                           control.extra == {"Tag": tag})
    return failures


def main() -> int:
    print("\nPhase 217: native forcing aliases in generic and deck schedules")
    failures = validate_simple_schedule_native_aliases()
    failures += validate_deck_schedule_native_aliases()
    if failures:
        print(f"\nPhase 217 failed ({failures} checks).")
        return 1
    print("\nGeneric and deck schedules now pass through native forcing aliases consistently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
