"""Phase 214: case-insensitive forcing container keys."""

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


def validate_boundary_container_aliases() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    q = float(case["bc"].value[:2].sum())
    bc = boundary_condition_from_control(
        case["grid"],
        {
            "BoundaryConditions": [
                {"TYPE": "RATE", "SIDE": "xmin", "Q": q, "COMPI": [0.2, 0.7, 0.1]},
                {"TYPE": "PRES", "SIDE": "xmax", "PRES": case["min_pressure"]},
            ]
        },
    )
    bc_records = boundary_condition_from_control(
        case["grid"],
        {
            "BCRecords": [
                {"TYPE": "RATE", "SIDE": "xmin", "Q": q, "COMPI": [0.3, 0.6, 0.1]},
            ]
        },
    )

    failures = 0
    failures += _check("BoundaryConditions container is parsed",
                       bc is not None
                       and bc.face.size == case["bc"].face.size
                       and bc.type[0] == "flux"
                       and bc.type[-1] == "pressure")
    failures += _check("BoundaryConditions records preserve components",
                       bc is not None
                       and bc.components.shape[1] == 3
                       and np.allclose(bc.components[0], [0.2, 0.7, 0.1]))
    failures += _check("BCRecords container is parsed",
                       bc_records is not None
                       and bc_records.type[0] == "flux"
                       and np.allclose(bc_records.components[0], [0.3, 0.6, 0.1]))
    return failures


def validate_source_container_aliases() -> int:
    grid, _, _, _, _, model, _ = make_line_drive_case(nx=4)
    rate = float(pore_volume(grid, model.rock).sum() / (1000.0 * DAY))
    src_terms = source_from_control(
        {
            "SourceTerms": [
                {"CELL": 0, "Q": rate, "COMPOSITION_I": [0.1, 0.8, 0.1]},
                {"CELL": grid.cells.num - 1, "MassRate": -rate},
            ]
        },
    )
    src_records = source_from_control(
        {
            "SrcRecords": [
                {"CELL": 0, "Q": rate, "ZI": [0.8, 0.1, 0.1]},
            ]
        },
    )

    failures = 0
    failures += _check("SourceTerms container is parsed",
                       src_terms is not None
                       and src_terms.cell.tolist() == [0, grid.cells.num - 1]
                       and np.allclose(src_terms.rate, [rate, -rate]))
    failures += _check("SourceTerms records preserve components",
                       src_terms is not None
                       and src_terms.components.shape == (2, 3)
                       and np.allclose(src_terms.components[0], [0.1, 0.8, 0.1]))
    failures += _check("SrcRecords container is parsed",
                       src_records is not None
                       and src_records.cell.tolist() == [0]
                       and np.allclose(src_records.components[0], [0.8, 0.1, 0.1]))
    return failures


def validate_mixed_forcing_container_aliases() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    qbc = float(case["bc"].value[:2].sum())
    qsrc = float(pore_volume(case["grid"], case["overall_model"].rock).sum() / (5000.0 * DAY))
    converted = schedule_controls_from_forcing_controls(
        case["grid"],
        [
            {
                "BoundaryConditions": [
                    {"TYPE": "RATE", "SIDE": "xmin", "Q": qbc, "COMPI": [0.4, 0.5, 0.1]},
                ],
                "SourceTerms": [
                    {"CELL": 0, "MassRate": qsrc, "ZI": [0.8, 0.1, 0.1]},
                ],
                "Label": "container-aliases",
            }
        ],
    )[0]

    failures = 0
    failures += _check("mixed container aliases build native forcing blocks",
                       converted.bc is not None
                       and converted.src is not None
                       and np.isclose(converted.src.rate[0], qsrc))
    failures += _check("mixed container aliases are consumed from extras",
                       converted.extra == {"Label": "container-aliases"})
    return failures


def main() -> int:
    print("\nPhase 214: case-insensitive forcing container keys")
    failures = validate_boundary_container_aliases()
    failures += validate_source_container_aliases()
    failures += validate_mixed_forcing_container_aliases()
    if failures:
        print(f"\nPhase 214 failed ({failures} checks).")
        return 1
    print("\nForcing converters now accept MATLAB-style mixed-case container keys.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
