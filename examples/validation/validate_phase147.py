"""Phase 147: MRST-like compositional boundary-control schedules."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    boundary_condition_from_control,
    run_schedule,
    schedule_controls_from_boundary_controls,
    simple_schedule_from_boundary_controls,
)
from pymrst.modules.compositional.examples.compositional_boundary_conditions_example import (  # noqa: E402
    make_boundary_conditions_case,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def validate_side_based_boundary_records() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    q = float(case["bc"].value[:2].sum())
    control = {
        "boundary_conditions": [
            {
                "type": "flux",
                "side": "xmin",
                "value": q,
                "sat": [1.0, 0.0],
                "components": [0.1, 0.9, 0.0],
            },
            {
                "type": "pressure",
                "side": "xmax",
                "value": case["min_pressure"],
                "sat": [1.0, 0.0],
                "components": [0.0, 0.0, 0.0],
            },
        ]
    }
    bc = boundary_condition_from_control(case["grid"], control)

    failures = 0
    failures += _check("side records create flux and pressure faces",
                       bc is not None
                       and bc.face.size == case["bc"].face.size
                       and bc.type == ["flux", "flux", "pressure", "pressure"])
    failures += _check("side flux is area-distributed like fluxside",
                       np.isclose(bc.value[:2].sum(), q)
                       and np.allclose(bc.value[:2], case["bc"].value[:2]))
    failures += _check("side components are broadcast per face",
                       bc.components.shape == (bc.face.size, 3)
                       and np.allclose(bc.components[0], [0.1, 0.9, 0.0]))
    return failures


def validate_explicit_face_boundary_records() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    left = case["bc"].face[:2]
    right = case["bc"].face[2:]
    control = {
        "BC": [
            {
                "type": "flux",
                "faces": left,
                "value": case["bc"].value[:2] * 0.5,
                "sat": [1.0, 0.0],
                "components": [0.7, 0.3, 0.0],
            },
            {
                "type": "pressure",
                "faces": right,
                "value": case["min_pressure"] * 0.9,
                "sat": [1.0, 0.0],
                "components": [0.0, 0.0, 0.0],
            },
        ]
    }
    bc = boundary_condition_from_control(case["grid"], control)

    failures = 0
    failures += _check("explicit face records preserve selected faces",
                       np.array_equal(bc.face, case["bc"].face))
    failures += _check("explicit face records preserve per-face values",
                       np.allclose(bc.value[:2], case["bc"].value[:2] * 0.5)
                       and np.allclose(bc.value[2:], case["min_pressure"] * 0.9))
    failures += _check("explicit face components stay aligned with records",
                       np.allclose(bc.components[0], [0.7, 0.3, 0.0])
                       and np.allclose(bc.components[-1], [0.0, 0.0, 0.0]))
    return failures


def validate_schedule_switching_from_boundary_controls() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    q = float(case["bc"].value[:2].sum())
    controls = [
        {
            "boundary_conditions": [
                {"type": "flux", "side": "xmin", "value": q, "sat": [1.0, 0.0], "components": [0.1, 0.9, 0.0]},
                {"type": "pressure", "side": "xmax", "value": case["min_pressure"], "sat": [1.0, 0.0]},
            ],
            "label": "CO2-rich",
        },
        {
            "boundary_conditions": [
                {"type": "flux", "side": "xmin", "value": q, "sat": [1.0, 0.0], "components": [0.8, 0.2, 0.0]},
                {"type": "pressure", "side": "xmax", "value": case["min_pressure"], "sat": [1.0, 0.0]},
            ],
            "label": "methane-rich",
        },
    ]
    converted = schedule_controls_from_boundary_controls(case["grid"], controls)
    schedule = simple_schedule_from_boundary_controls(
        case["schedule"].step.val[:2],
        case["grid"],
        controls,
        step_control=[1, 2],
    )
    result = run_schedule(case["overall_model"], case["initial_state"], schedule)

    failures = 0
    failures += _check("boundary-control conversion preserves extras",
                       converted[0].extra == {"label": "CO2-rich"}
                       and converted[1].extra == {"label": "methane-rich"})
    failures += _check("schedule keeps one-based switched controls",
                       schedule.step.control.tolist() == [1, 2])
    failures += _check("second control carries changed component stream",
                       np.allclose(schedule.control[1].bc.components[0], [0.8, 0.2, 0.0]))
    failures += _check("switched boundary stream affects the inlet composition",
                       result.final_state.z[0, 0] > case["initial_state"].z[0, 0])
    failures += _check("model boundary condition is restored after run",
                       case["overall_model"].bc is case["bc"])
    return failures


def main() -> int:
    print("\nPhase 147: MRST-like compositional boundary-control schedules")
    failures = validate_side_based_boundary_records()
    failures += validate_explicit_face_boundary_records()
    failures += validate_schedule_switching_from_boundary_controls()
    if failures:
        print(f"\nPhase 147 failed ({failures} checks).")
        return 1
    print("\nBoundary-control dictionaries now build switched compositional BC schedules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
