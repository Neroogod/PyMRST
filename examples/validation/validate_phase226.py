"""Phase 226: MRST-style psideh controls in compositional schedules."""

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


class TwoPhaseFluid:
    def properties(self, state):
        return np.array([1.0, 2.0]), np.array([100.0, 50.0])

    def relperm(self, sat):
        return np.asarray(sat, dtype=float)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _omega(sat) -> float:
    mob = np.asarray(sat, dtype=float) / np.array([1.0, 2.0])
    return float(np.sum(mob * np.array([100.0, 50.0])) / np.sum(mob))


def validate_direct_psideh_control() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    grid = case["grid"]
    sat = np.array([0.2, 0.8])
    control = {
        "pSideH": "xmax",
        "fluid": TwoPhaseFluid(),
        "value": 1234.0,
        "sat": sat,
        "COMPI": [0.1, 0.9, 0.0],
        "i1": [1],
        "range": [3],
        "gravity": [10.0, 0.0, 0.0],
    }
    bc = boundary_condition_from_control(grid, control)
    expected = 1234.0 + _omega(sat) * (grid.faces.centroids[bc.face, 0] * 10.0)

    failures = 0
    failures += _check("pSideH creates pressure boundaries",
                       bc is not None and set(bc.type) == {"pressure"})
    failures += _check("pSideH respects i1/range restrictions",
                       bc.face.size == 1)
    failures += _check("pSideH value becomes reference pressure",
                       np.allclose(bc.value, expected, rtol=1e-13, atol=1e-10))
    failures += _check("pSideH keeps components aligned with faces",
                       bc.components.shape == (bc.face.size, 3)
                       and np.allclose(bc.components, [0.1, 0.9, 0.0]))
    return failures


def validate_schedule_cleanup() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    fluid = TwoPhaseFluid()
    boundary = schedule_controls_from_boundary_controls(
        case["grid"],
        [
            {
                "psideh": "xmax",
                "fluid": fluid,
                "refPressure": 100.0,
                "refPosition": [0.0, 0.0, 0.0],
                "gravity": [0.0, 0.0, 9.80665],
                "sat": [1.0, 0.0],
                "COMPI": [1.0, 0.0, 0.0],
                "Label": "boundary-psideh",
            }
        ],
    )[0]
    mixed = schedule_controls_from_forcing_controls(
        case["grid"],
        [
            {
                "HydrostaticSide": "xmin",
                "Fluid": fluid,
                "pn": 90.0,
                "refDepth": 0.0,
                "Gravity": [0.0, 0.0, 9.80665],
                "COMPI": [0.0, 1.0, 0.0],
                "Label": "mixed-psideh",
            }
        ],
    )[0]

    failures = 0
    failures += _check("boundary schedule consumes psideh alias",
                       boundary.bc is not None and set(boundary.bc.type) == {"pressure"})
    failures += _check("boundary schedule filters psideh options",
                       boundary.extra == {"Label": "boundary-psideh"})
    failures += _check("mixed schedule consumes HydrostaticSide alias",
                       mixed.bc is not None and set(mixed.bc.type) == {"pressure"})
    failures += _check("mixed schedule filters psideh options",
                       mixed.extra == {"Label": "mixed-psideh"})
    return failures


def validate_error_paths() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    failures = 0
    try:
        boundary_condition_from_control(case["grid"], {"psideh": "xmax", "value": 100.0})
    except ValueError:
        failures += _check("psideh controls require fluid", True)
    else:
        failures += _check("psideh controls require fluid", False)

    try:
        boundary_condition_from_control(
            case["grid"],
            {"psideh": "xmax", "fluxSide": "xmin", "fluid": TwoPhaseFluid(), "value": 100.0},
        )
    except ValueError:
        failures += _check("conflicting typed side aliases fail loudly", True)
    else:
        failures += _check("conflicting typed side aliases fail loudly", False)
    return failures


def main() -> int:
    print("\nPhase 226: MRST-style psideh controls in compositional schedules")
    failures = validate_direct_psideh_control()
    failures += validate_schedule_cleanup()
    failures += validate_error_paths()
    if failures:
        print(f"\nPhase 226 failed ({failures} checks).")
        return 1
    print("\nCompositional forcing controls now accept MRST-style psideh hydrostatic boundaries.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
