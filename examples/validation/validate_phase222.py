"""Phase 222: boundary side-name aliases."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    boundary_condition_from_control,
    schedule_controls_from_boundary_controls,
    schedule_controls_from_forcing_controls,
)
from pymrst.modules.compositional.examples.compositional_boundary_conditions_example import (  # noqa: E402
    make_boundary_conditions_case,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def validate_direct_side_aliases() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    bc_face_side = boundary_condition_from_control(
        case["grid"],
        {"faceSide": "xmin", "Q": 0.0, "COMPI": [1.0, 0.0, 0.0]},
    )
    bc_side_name = boundary_condition_from_control(
        case["grid"],
        {"sideName": "xmax", "PRES": 100.0, "COMPI": [0.0, 1.0, 0.0]},
    )

    failures = 0
    failures += _check("faceSide alias creates a flux boundary",
                       bc_face_side is not None
                       and bc_face_side.face.size > 0
                       and set(bc_face_side.type) == {"flux"}
                       and bc_face_side.components.shape[1] == 3)
    failures += _check("sideName alias creates a pressure boundary",
                       bc_side_name is not None
                       and bc_side_name.face.size > 0
                       and set(bc_side_name.type) == {"pressure"}
                       and np.allclose(bc_side_name.value, 100.0))
    return failures


def validate_schedule_side_alias_cleanup() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    boundary = schedule_controls_from_boundary_controls(
        case["grid"],
        [
            {
                "face_side": "xmin",
                "Q": 0.0,
                "COMPI": [1.0, 0.0, 0.0],
                "Label": "boundary-side-alias",
            }
        ],
    )[0]
    mixed = schedule_controls_from_forcing_controls(
        case["grid"],
        [
            {
                "SideName": "xmax",
                "PRES": 100.0,
                "COMPI": [0.0, 1.0, 0.0],
                "Label": "mixed-side-alias",
            }
        ],
    )[0]

    failures = 0
    failures += _check("boundary schedule consumes face_side alias",
                       boundary.bc is not None and set(boundary.bc.type) == {"flux"})
    failures += _check("boundary schedule filters face_side alias",
                       boundary.extra == {"Label": "boundary-side-alias"})
    failures += _check("mixed schedule consumes SideName alias",
                       mixed.bc is not None and set(mixed.bc.type) == {"pressure"})
    failures += _check("mixed schedule filters SideName alias",
                       mixed.extra == {"Label": "mixed-side-alias"})
    return failures


def main() -> int:
    print("\nPhase 222: boundary side-name aliases")
    failures = validate_direct_side_aliases()
    failures += validate_schedule_side_alias_cleanup()
    if failures:
        print(f"\nPhase 222 failed ({failures} checks).")
        return 1
    print("\nBoundary controls now accept faceSide/sideName aliases consistently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
