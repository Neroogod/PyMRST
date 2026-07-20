"""Phase 211: MRST-style boundary type/value aliases for forcing controls."""

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


def validate_mapping_aliases() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    q = float(case["bc"].value[:2].sum())
    bc = boundary_condition_from_control(
        case["grid"],
        {
            "BC": [
                {"kind": "rate", "side": "xmin", "q": q, "sat": [1.0, 0.0], "compi": [0.7, 0.2, 0.1]},
                {"bc_type": "pres", "side": "xmax", "pres": case["min_pressure"], "sat": [0.5, 0.5]},
            ]
        },
    )

    failures = 0
    failures += _check("mapping rate/q aliases become flux boundaries",
                       bc is not None
                       and bc.type[:2] == ["flux", "flux"]
                       and np.isclose(float(bc.value[:2].sum()), q))
    failures += _check("mapping pres alias becomes pressure boundaries",
                       bc.type[2:] == ["pressure", "pressure"]
                       and np.allclose(bc.value[2:], case["min_pressure"]))
    failures += _check("aliased mapping components are preserved",
                       bc.components.shape == (bc.face.size, 3)
                       and np.allclose(bc.components[0], [0.7, 0.2, 0.1]))
    return failures


def validate_sequence_aliases() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    q = float(case["bc"].value[:2].sum())
    bc = boundary_condition_from_control(
        case["grid"],
        {
            "bc": [
                ["rate", "xmin", q, [1.0, 0.0], [0.4, 0.5, 0.1]],
                ["pres", "xmax", case["min_pressure"], [0.5, 0.5], [0.0, 0.0, 0.0]],
            ]
        },
    )

    failures = 0
    failures += _check("sequence rate alias becomes flux side record",
                       bc is not None and bc.type[:2] == ["flux", "flux"])
    failures += _check("sequence pres alias becomes pressure side record",
                       bc.type[2:] == ["pressure", "pressure"])
    failures += _check("sequence aliases keep component rows aligned",
                       np.allclose(bc.components[0], [0.4, 0.5, 0.1])
                       and np.allclose(bc.components[-1], [0.0, 0.0, 0.0]))
    return failures


def validate_face_table_aliases_and_metadata_consumption() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    faces = case["bc"].face
    control = {
        "face": faces,
        "type": ["rate", "neumann", "pres", "dirichlet"],
        "value": [case["bc"].value[0], case["bc"].value[1],
                  case["min_pressure"], 0.95 * case["min_pressure"]],
        "compi": [
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        "label": "alias-table",
    }
    bc = boundary_condition_from_control(case["grid"], control)
    converted = schedule_controls_from_forcing_controls(case["grid"], [control])[0]

    failures = 0
    failures += _check("face-table type aliases normalize per row",
                       bc is not None
                       and bc.type == ["flux", "flux", "pressure", "pressure"])
    failures += _check("face-table alias conversion preserves values",
                       np.allclose(bc.value, control["value"]))
    failures += _check("face-table alias metadata is consumed except unrelated fields",
                       converted.extra == {"label": "alias-table"})
    return failures


def main() -> int:
    print("\nPhase 211: MRST-style boundary type/value aliases")
    failures = validate_mapping_aliases()
    failures += validate_sequence_aliases()
    failures += validate_face_table_aliases_and_metadata_consumption()
    if failures:
        print(f"\nPhase 211 failed ({failures} checks).")
        return 1
    print("\nBoundary forcing controls now accept MRST-style type/value aliases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
