"""Phase 155: MRST-style composition aliases for forcing controls."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    boundary_condition_from_control,
    make_line_drive_case,
    schedule_controls_from_forcing_controls,
    source_from_control,
)
from pymrst.modules.compositional.examples.compositional_boundary_conditions_example import (  # noqa: E402
    make_boundary_conditions_case,
)
from pymrst.params.rock import pore_volume  # noqa: E402
from pymrst.utils.units import DAY  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def validate_boundary_compi_alias() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    q = float(case["bc"].value[:2].sum())
    bc = boundary_condition_from_control(
        case["grid"],
        {
            "bc": [
                {"type": "flux", "side": "xmin", "value": q, "sat": [1.0, 0.0], "compi": [0.2, 0.7, 0.1]},
                {"type": "pressure", "side": "xmax", "value": case["min_pressure"], "sat": [1.0, 0.0]},
            ]
        },
    )

    failures = 0
    failures += _check("boundary compi alias is parsed as components",
                       bc is not None
                       and bc.components.shape == (bc.face.size, 3)
                       and np.allclose(bc.components[0], [0.2, 0.7, 0.1]))
    failures += _check("boundary records without alias receive zero component rows",
                       np.allclose(bc.components[-1], [0.0, 0.0, 0.0]))
    return failures


def validate_source_composition_i_alias() -> int:
    grid, _, _, _, _, model, _ = make_line_drive_case(nx=4)
    model.wells = []
    rate = float(pore_volume(grid, model.rock).sum() / (1000.0 * DAY))
    src = source_from_control(
        {
            "SRC": [
                {"cell": 0, "rate": rate, "sat": [0.0, 1.0], "composition_i": [0.3, 0.6, 0.1]},
                {"cell": grid.cells.num - 1, "rate": -rate, "sat": [0.5, 0.5], "zi": [0.0, 0.0, 0.0]},
            ]
        }
    )

    failures = 0
    failures += _check("source composition_i alias is parsed as components",
                       src is not None
                       and src.components.shape == (2, 3)
                       and np.allclose(src.components[0], [0.3, 0.6, 0.1]))
    failures += _check("source zi alias is parsed as components",
                       np.allclose(src.components[1], [0.0, 0.0, 0.0]))
    return failures


def validate_mixed_control_aliases_are_not_extra_metadata() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    qbc = float(case["bc"].value[:2].sum())
    qsrc = float(pore_volume(case["grid"], case["overall_model"].rock).sum() / (5000.0 * DAY))
    converted = schedule_controls_from_forcing_controls(
        case["grid"],
        [
            {
                "bc": [{"type": "flux", "side": "xmin", "value": qbc, "compi": [0.4, 0.5, 0.1]}],
                "src": [{"cell": 0, "rate": qsrc, "composition_i": [0.8, 0.1, 0.1]}],
                "label": "alias-control",
            }
        ],
    )[0]

    failures = 0
    failures += _check("mixed forcing aliases reach native BC and source components",
                       np.allclose(converted.bc.components[0], [0.4, 0.5, 0.1])
                       and np.allclose(converted.src.components[0], [0.8, 0.1, 0.1]))
    failures += _check("mixed forcing aliases are consumed, not stored as extras",
                       converted.extra == {"label": "alias-control"})
    return failures


def main() -> int:
    print("\nPhase 155: MRST-style composition aliases for forcing controls")
    failures = validate_boundary_compi_alias()
    failures += validate_source_composition_i_alias()
    failures += validate_mixed_control_aliases_are_not_extra_metadata()
    if failures:
        print(f"\nPhase 155 failed ({failures} checks).")
        return 1
    print("\nForcing controls now accept MRST-style composition aliases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
