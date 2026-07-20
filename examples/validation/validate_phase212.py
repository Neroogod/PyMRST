"""Phase 212: MRST-style source rate aliases for forcing controls."""

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
    schedule_controls_from_source_controls,
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


def _source_case():
    grid, _, _, _, _, model, _ = make_line_drive_case(nx=4)
    model.wells = []
    rate = float(pore_volume(grid, model.rock).sum() / (1000.0 * DAY))
    return grid, model, rate


def validate_direct_source_aliases() -> int:
    grid, _, rate = _source_case()
    src = source_from_control(
        {
            "sources": [
                {"cell": 0, "q": rate, "sat": [0.0, 1.0], "components": [0.7, 0.2, 0.1]},
                {"cell": grid.cells.num - 1, "massrate": -rate, "sat": [0.5, 0.5]},
            ]
        }
    )

    failures = 0
    failures += _check("source q alias is parsed as rate",
                       src is not None and np.isclose(src.rate[0], rate))
    failures += _check("source massrate alias is parsed as rate",
                       src is not None and np.isclose(src.rate[1], -rate))
    failures += _check("source aliases preserve component rows",
                       src is not None
                       and src.components.shape == (2, 3)
                       and np.allclose(src.components[0], [0.7, 0.2, 0.1]))
    return failures


def validate_source_schedule_alias_metadata() -> int:
    grid, _, rate = _source_case()
    controls = [
        {"cell": 0, "q": rate, "label": "q-source"},
        {"cell": grid.cells.num - 1, "mass_rate": -rate, "label": "mass-source"},
    ]
    converted = schedule_controls_from_source_controls(controls)

    failures = 0
    failures += _check("source schedule q alias is consumed",
                       np.isclose(converted[0].src.rate[0], rate)
                       and converted[0].extra == {"label": "q-source"})
    failures += _check("source schedule mass_rate alias is consumed",
                       np.isclose(converted[1].src.rate[0], -rate)
                       and converted[1].extra == {"label": "mass-source"})
    failures += _check("source aliases keep cell indices",
                       converted[0].src.cell.tolist() == [0]
                       and converted[1].src.cell.tolist() == [grid.cells.num - 1])
    return failures


def validate_mixed_forcing_source_aliases() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    qbc = float(case["bc"].value[:2].sum())
    qsrc = float(pore_volume(case["grid"], case["overall_model"].rock).sum() / (5000.0 * DAY))
    converted = schedule_controls_from_forcing_controls(
        case["grid"],
        [
            {
                "bc": [{"type": "flux", "side": "xmin", "q": qbc, "compi": [0.4, 0.5, 0.1]}],
                "src": [{"cell": 0, "massrate": qsrc, "composition_i": [0.8, 0.1, 0.1]}],
                "label": "mixed-source-alias",
            }
        ],
    )[0]

    failures = 0
    failures += _check("mixed forcing source massrate alias reaches native source",
                       np.isclose(converted.src.rate[0], qsrc))
    failures += _check("mixed forcing aliases are consumed from metadata",
                       converted.extra == {"label": "mixed-source-alias"})
    return failures


def validate_source_aliases_are_not_boundaries() -> int:
    grid, _, rate = _source_case()
    for record in (
        {"cell": 0, "q": rate},
        {"cell": 0, "massrate": rate},
        {"cell": 0, "mass_rate": rate},
    ):
        if boundary_condition_from_control(grid, record) is not None:
            return _check("source rate aliases do not create boundary records", False)
    return _check("source rate aliases do not create boundary records", True)


def main() -> int:
    print("\nPhase 212: MRST-style source rate aliases")
    failures = validate_direct_source_aliases()
    failures += validate_source_schedule_alias_metadata()
    failures += validate_mixed_forcing_source_aliases()
    failures += validate_source_aliases_are_not_boundaries()
    if failures:
        print(f"\nPhase 212 failed ({failures} checks).")
        return 1
    print("\nSource controls now accept MRST-style q/massrate aliases.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
