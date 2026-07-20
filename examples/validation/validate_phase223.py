"""Phase 223: boundary typed-side aliases."""

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


def validate_typed_side_aliases() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    pressure = boundary_condition_from_control(
        case["grid"],
        {"pressureSide": "xmax", "value": 100.0, "COMPI": [0.0, 1.0, 0.0]},
    )
    flux = boundary_condition_from_control(
        case["grid"],
        {"FluxSide": "xmin", "value": 0.0, "COMPI": [1.0, 0.0, 0.0]},
    )

    failures = 0
    failures += _check("pressureSide infers pressure boundary",
                       pressure is not None
                       and pressure.face.size > 0
                       and set(pressure.type) == {"pressure"}
                       and np.allclose(pressure.value, 100.0))
    failures += _check("fluxSide infers flux boundary",
                       flux is not None
                       and flux.face.size > 0
                       and set(flux.type) == {"flux"}
                       and flux.components.shape[1] == 3)
    return failures


def validate_typed_side_schedule_cleanup() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    boundary = schedule_controls_from_boundary_controls(
        case["grid"],
        [
            {
                "pressure_side": "xmax",
                "value": 100.0,
                "COMPI": [0.0, 1.0, 0.0],
                "Label": "boundary-typed-side",
            }
        ],
    )[0]
    mixed = schedule_controls_from_forcing_controls(
        case["grid"],
        [
            {
                "fluxSide": "xmin",
                "value": 0.0,
                "COMPI": [1.0, 0.0, 0.0],
                "Label": "mixed-typed-side",
            }
        ],
    )[0]

    failures = 0
    failures += _check("boundary schedule consumes pressure_side alias",
                       boundary.bc is not None and set(boundary.bc.type) == {"pressure"})
    failures += _check("boundary schedule filters pressure_side alias",
                       boundary.extra == {"Label": "boundary-typed-side"})
    failures += _check("mixed schedule consumes fluxSide alias",
                       mixed.bc is not None and set(mixed.bc.type) == {"flux"})
    failures += _check("mixed schedule filters fluxSide alias",
                       mixed.extra == {"Label": "mixed-typed-side"})
    return failures


def validate_conflicting_typed_side() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    try:
        boundary_condition_from_control(
            case["grid"],
            {"pressureSide": "xmax", "type": "flux", "value": 0.0},
        )
    except ValueError as err:
        conflict = "conflicts" in str(err)
    else:
        conflict = False
    return _check("typed side conflicts fail loudly", conflict)


def main() -> int:
    print("\nPhase 223: boundary typed-side aliases")
    failures = validate_typed_side_aliases()
    failures += validate_typed_side_schedule_cleanup()
    failures += validate_conflicting_typed_side()
    if failures:
        print(f"\nPhase 223 failed ({failures} checks).")
        return 1
    print("\nBoundary controls now accept pressureSide/fluxSide aliases consistently.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
