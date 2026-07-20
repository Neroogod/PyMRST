"""Phase 224: MRST pside boundary alias."""

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


def validate_pside_alias() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    lower = boundary_condition_from_control(
        case["grid"],
        {"pside": "xmax", "value": 100.0, "COMPI": [0.0, 1.0, 0.0]},
    )
    camel = boundary_condition_from_control(
        case["grid"],
        {"pSide": "xmin", "value": 90.0, "COMPI": [1.0, 0.0, 0.0]},
    )

    failures = 0
    failures += _check("pside infers pressure boundary",
                       lower is not None
                       and lower.face.size > 0
                       and set(lower.type) == {"pressure"}
                       and np.allclose(lower.value, 100.0))
    failures += _check("pSide infers pressure boundary",
                       camel is not None
                       and camel.face.size > 0
                       and set(camel.type) == {"pressure"}
                       and np.allclose(camel.value, 90.0))
    return failures


def validate_pside_schedule_cleanup() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    boundary = schedule_controls_from_boundary_controls(
        case["grid"],
        [
            {
                "pside": "xmax",
                "value": 100.0,
                "COMPI": [0.0, 1.0, 0.0],
                "Label": "boundary-pside",
            }
        ],
    )[0]
    mixed = schedule_controls_from_forcing_controls(
        case["grid"],
        [
            {
                "PSide": "xmin",
                "value": 90.0,
                "COMPI": [1.0, 0.0, 0.0],
                "Label": "mixed-pside",
            }
        ],
    )[0]

    failures = 0
    failures += _check("boundary schedule consumes pside alias",
                       boundary.bc is not None and set(boundary.bc.type) == {"pressure"})
    failures += _check("boundary schedule filters pside alias",
                       boundary.extra == {"Label": "boundary-pside"})
    failures += _check("mixed schedule consumes PSide alias",
                       mixed.bc is not None and set(mixed.bc.type) == {"pressure"})
    failures += _check("mixed schedule filters PSide alias",
                       mixed.extra == {"Label": "mixed-pside"})
    return failures


def main() -> int:
    print("\nPhase 224: MRST pside boundary alias")
    failures = validate_pside_alias()
    failures += validate_pside_schedule_cleanup()
    if failures:
        print(f"\nPhase 224 failed ({failures} checks).")
        return 1
    print("\nBoundary controls now accept MRST pside/pSide aliases consistently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
