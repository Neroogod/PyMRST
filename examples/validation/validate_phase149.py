"""Phase 149: mixed MRST-like boundary/source forcing controls."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    run_schedule,
    schedule_controls_from_forcing_controls,
    simple_schedule_from_forcing_controls,
)
from pymrst.modules.compositional.examples.compositional_boundary_conditions_example import (  # noqa: E402
    make_boundary_conditions_case,
)
from pymrst.params.rock import pore_volume  # noqa: E402
from pymrst.utils.units import DAY  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _combined_controls(case):
    grid = case["grid"]
    qbc = float(case["bc"].value[:2].sum())
    qsrc = float(pore_volume(grid, case["overall_model"].rock).sum() / (5000.0 * DAY))
    return [
        {
            "bc": [
                {"type": "flux", "side": "xmin", "value": qbc, "sat": [1.0, 0.0], "components": [0.1, 0.9, 0.0]},
                {"type": "pressure", "side": "xmax", "value": case["min_pressure"], "sat": [1.0, 0.0]},
            ],
            "src": [
                {"cell": 0, "rate": qsrc, "sat": [0.0, 1.0], "components": [0.9, 0.05, 0.05]},
                {"cell": grid.cells.num - 1, "rate": -qsrc, "sat": [0.5, 0.5]},
            ],
            "label": "bc-and-source-a",
        },
        {
            "bc": [
                {"type": "flux", "side": "xmin", "value": qbc, "sat": [1.0, 0.0], "components": [0.8, 0.2, 0.0]},
                {"type": "pressure", "side": "xmax", "value": case["min_pressure"], "sat": [1.0, 0.0]},
            ],
            "src": [
                {"cell": 0, "rate": qsrc, "sat": [0.0, 1.0], "components": [0.05, 0.9, 0.05]},
                {"cell": grid.cells.num - 1, "rate": -qsrc, "sat": [0.5, 0.5]},
            ],
            "label": "bc-and-source-b",
        },
    ]


def validate_mixed_control_conversion() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    controls = _combined_controls(case)
    converted = schedule_controls_from_forcing_controls(case["grid"], controls)

    failures = 0
    failures += _check("mixed controls produce native BC and source blocks",
                       converted[0].bc is not None
                       and converted[0].src is not None
                       and converted[0].bc.face.size == case["bc"].face.size
                       and converted[0].src.cell.tolist() == [0, case["grid"].cells.num - 1])
    failures += _check("lowercase bc/src records are parsed, not copied raw",
                       hasattr(converted[0].bc, "components")
                       and hasattr(converted[0].src, "components")
                       and np.allclose(converted[0].bc.components[0], [0.1, 0.9, 0.0])
                       and np.allclose(converted[0].src.components[0], [0.9, 0.05, 0.05]))
    failures += _check("mixed conversion preserves unrelated metadata",
                       converted[0].extra == {"label": "bc-and-source-a"}
                       and converted[1].extra == {"label": "bc-and-source-b"})
    return failures


def validate_source_only_control_is_not_boundary() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    qsrc = float(pore_volume(case["grid"], case["overall_model"].rock).sum() / (5000.0 * DAY))
    converted = schedule_controls_from_forcing_controls(
        case["grid"],
        [{"cell": 0, "rate": qsrc, "sat": [0.0, 1.0], "components": [0.9, 0.05, 0.05]}],
    )

    failures = 0
    failures += _check("source-only top-level record does not become a BC",
                       converted[0].bc is None and converted[0].src is not None)
    failures += _check("source-only top-level record keeps its source cell",
                       converted[0].src.cell.tolist() == [0])
    return failures


def validate_mixed_control_schedule_run() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    controls = _combined_controls(case)
    schedule = simple_schedule_from_forcing_controls(
        case["schedule"].step.val[:2],
        case["grid"],
        controls,
        step_control=[1, 2],
    )
    result = run_schedule(case["overall_model"], case["initial_state"], schedule)

    failures = 0
    failures += _check("mixed forcing schedule keeps one-based controls",
                       schedule.step.control.tolist() == [1, 2])
    failures += _check("second control changes both BC and source streams",
                       np.allclose(schedule.control[1].bc.components[0], [0.8, 0.2, 0.0])
                       and np.allclose(schedule.control[1].src.components[0], [0.05, 0.9, 0.05]))
    failures += _check("mixed forcing schedule advances finite states",
                       np.isfinite(result.final_state.pressure).all()
                       and np.isfinite(result.final_state.z).all())
    failures += _check("model BC and source are restored after mixed schedule",
                       case["overall_model"].bc is case["bc"]
                       and getattr(case["overall_model"], "src", None) is None)
    return failures


def main() -> int:
    print("\nPhase 149: mixed MRST-like boundary/source forcing controls")
    failures = validate_mixed_control_conversion()
    failures += validate_source_only_control_is_not_boundary()
    failures += validate_mixed_control_schedule_run()
    if failures:
        print(f"\nPhase 149 failed ({failures} checks).")
        return 1
    print("\nMixed boundary/source controls now build native compositional schedules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
