"""Phase 156: MRST-style table boundary records for forcing controls."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    boundary_condition_from_control,
    schedule_controls_from_forcing_controls,
)
from pymrst.modules.compositional.examples.compositional_boundary_conditions_example import (  # noqa: E402
    make_boundary_conditions_case,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def validate_mixed_type_face_table() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    faces = case["bc"].face
    values = np.array([case["bc"].value[0], case["bc"].value[1],
                       case["min_pressure"], case["min_pressure"] * 0.95])
    control = {
        "bc": {
            "face": faces,
            "type": ["flux", "flux", "pressure", "pressure"],
            "value": values,
            "sat": [[1.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.5, 0.5]],
            "compi": [
                [0.1, 0.8, 0.1],
                [0.2, 0.7, 0.1],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
        }
    }
    bc = boundary_condition_from_control(case["grid"], control)

    failures = 0
    failures += _check("face table preserves row order",
                       bc is not None and np.array_equal(bc.face, faces))
    failures += _check("face table preserves per-row boundary types",
                       bc.type == ["flux", "flux", "pressure", "pressure"])
    failures += _check("face table preserves per-row values",
                       np.allclose(bc.value, values))
    failures += _check("face table preserves per-row saturations",
                       np.allclose(bc.sat[2], [0.5, 0.5]))
    failures += _check("face table preserves per-row composition aliases",
                       bc.components.shape == (faces.size, 3)
                       and np.allclose(bc.components[1], [0.2, 0.7, 0.1]))
    return failures


def validate_mixed_forcing_table_metadata() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    faces = case["bc"].face[:2]
    controls = [
        {
            "BC": {
                "face": faces,
                "type": ["flux", "pressure"],
                "value": [case["bc"].value[0], case["min_pressure"]],
                "compi": [[0.9, 0.1, 0.0], [0.0, 0.0, 0.0]],
            },
            "label": "table-bc",
        }
    ]
    converted = schedule_controls_from_forcing_controls(case["grid"], controls)
    extra = converted[0].extra

    failures = 0
    failures += _check("forcing controls accept nested BC table records",
                       converted[0].bc is not None
                       and converted[0].bc.type == ["flux", "pressure"])
    failures += _check("nested BC table aliases are consumed",
                       extra == {"label": "table-bc"})
    return failures


def main() -> int:
    print("\nPhase 156: MRST-style table boundary records for forcing controls")
    failures = validate_mixed_type_face_table()
    failures += validate_mixed_forcing_table_metadata()
    if failures:
        print(f"\nPhase 156 failed ({failures} checks).")
        return 1
    print("\nBoundary forcing controls now accept MRST-style face/type/value tables.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
