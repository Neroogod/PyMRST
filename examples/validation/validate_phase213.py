"""Phase 213: case-insensitive forcing metadata consumption."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    make_line_drive_case,
    schedule_controls_from_boundary_controls,
    schedule_controls_from_forcing_controls,
    schedule_controls_from_source_controls,
)
from pymrst.modules.compositional.examples.compositional_boundary_conditions_example import (  # noqa: E402
    make_boundary_conditions_case,
)
from pymrst.params.rock import pore_volume  # noqa: E402
from pymrst.utils.units import DAY  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def validate_boundary_metadata_filtering() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    q = float(case["bc"].value[:2].sum())
    converted = schedule_controls_from_boundary_controls(
        case["grid"],
        [
            {
                "TYPE": "RATE",
                "SIDE": "xmin",
                "Q": q,
                "COMPI": [0.2, 0.7, 0.1],
                "Label": "upper-bc",
            }
        ],
    )[0]

    failures = 0
    failures += _check("uppercase boundary aliases build a native BC",
                       converted.bc is not None
                       and converted.bc.type[0] == "flux"
                       and np.isclose(converted.bc.value[:2].sum(), q))
    failures += _check("uppercase boundary aliases are consumed from extras",
                       converted.extra == {"Label": "upper-bc"})
    return failures


def validate_source_metadata_filtering() -> int:
    grid, _, _, _, _, model, _ = make_line_drive_case(nx=4)
    rate = float(pore_volume(grid, model.rock).sum() / (1000.0 * DAY))
    converted = schedule_controls_from_source_controls(
        [
            {
                "CELL": 0,
                "MassRate": rate,
                "COMPOSITION_I": [0.1, 0.8, 0.1],
                "Label": "mixed-source",
            }
        ],
    )[0]

    failures = 0
    failures += _check("mixed-case source aliases build a native source",
                       converted.src is not None
                       and converted.src.cell.tolist() == [0]
                       and np.isclose(converted.src.rate[0], rate))
    failures += _check("mixed-case source aliases are consumed from extras",
                       converted.extra == {"Label": "mixed-source"})
    return failures


def validate_mixed_forcing_metadata_filtering() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    qbc = float(case["bc"].value[:2].sum())
    qsrc = float(pore_volume(case["grid"], case["overall_model"].rock).sum() / (5000.0 * DAY))
    converted = schedule_controls_from_forcing_controls(
        case["grid"],
        [
            {
                "BC": [{"TYPE": "RATE", "SIDE": "xmin", "Q": qbc, "COMPI": [0.4, 0.5, 0.1]}],
                "SRC": [{"CELL": 0, "MassRate": qsrc, "ZI": [0.8, 0.1, 0.1]}],
                "Label": "mixed-forcing",
            }
        ],
    )[0]

    failures = 0
    failures += _check("mixed forcing case-insensitive aliases reach native objects",
                       converted.bc is not None
                       and converted.src is not None
                       and np.isclose(converted.src.rate[0], qsrc))
    failures += _check("mixed forcing consumed aliases do not leak into extras",
                       converted.extra == {"Label": "mixed-forcing"})
    return failures


def main() -> int:
    print("\nPhase 213: case-insensitive forcing metadata consumption")
    failures = validate_boundary_metadata_filtering()
    failures += validate_source_metadata_filtering()
    failures += validate_mixed_forcing_metadata_filtering()
    if failures:
        print(f"\nPhase 213 failed ({failures} checks).")
        return 1
    print("\nForcing schedule converters now filter consumed aliases case-insensitively.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
