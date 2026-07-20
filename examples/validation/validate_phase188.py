"""Phase 188: full MRST-like schedule mapping conversion."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    run_schedule,
    simple_schedule_from_mrst_schedule,
)
from pymrst.modules.compositional.examples.compositional_boundary_conditions_example import (  # noqa: E402
    make_boundary_conditions_case,
)
from pymrst.params.rock import pore_volume  # noqa: E402
from pymrst.utils.units import DAY  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _mrst_like_schedule(case):
    grid = case["grid"]
    qbc = float(case["bc"].value[:2].sum())
    qsrc = float(pore_volume(grid, case["overall_model"].rock).sum() / (5000.0 * DAY))
    return {
        "step": {
            "val": np.asarray(case["schedule"].step.val[:3], dtype=float),
            "control": [1, 2, 1],
        },
        "control": [
            {
                "bc": [
                    {"type": "flux", "side": "xmin", "value": qbc, "sat": [1.0, 0.0],
                     "components": [0.1, 0.8, 0.1]},
                    {"type": "pressure", "side": "xmax", "value": case["min_pressure"], "sat": [1.0, 0.0]},
                ],
                "src": [
                    {"cell": 0, "rate": qsrc, "sat": [0.0, 1.0], "components": [0.8, 0.1, 0.1]},
                    {"cell": grid.cells.num - 1, "rate": -qsrc, "sat": [0.5, 0.5]},
                ],
                "name": "forcing-a",
            },
            {
                "BC": {
                    "face": case["bc"].face,
                    "type": ["flux", "flux", "pressure", "pressure"],
                    "value": [case["bc"].value[0], case["bc"].value[1],
                              case["min_pressure"] * 0.95, case["min_pressure"] * 0.95],
                    "sat": [[1.0, 0.0], [1.0, 0.0], [0.5, 0.5], [0.5, 0.5]],
                    "compi": [[0.75, 0.2, 0.05], [0.65, 0.3, 0.05],
                              [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                },
                "SRC": [
                    {"cell": 0, "value": qsrc, "sat": [0.0, 1.0], "zi": [0.05, 0.9, 0.05]},
                    {"cell": grid.cells.num - 1, "value": -qsrc, "sat": [0.5, 0.5]},
                ],
                "name": "forcing-b",
            },
        ],
    }


def validate_whole_schedule_mapping_conversion() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    native = simple_schedule_from_mrst_schedule(_mrst_like_schedule(case), grid=case["grid"])

    failures = 0
    failures += _check("whole schedule preserves report timesteps",
                       np.allclose(native.step.val, case["schedule"].step.val[:3]))
    failures += _check("whole schedule preserves one-based control indices",
                       native.step.control.tolist() == [1, 2, 1])
    failures += _check("first control converts nested list BC/SRC records",
                       native.control[0].bc is not None
                       and native.control[0].src is not None
                       and np.allclose(native.control[0].bc.components[0], [0.1, 0.8, 0.1])
                       and np.allclose(native.control[0].src.components[0], [0.8, 0.1, 0.1]))
    failures += _check("second control converts MRST-style BC table plus SRC records",
                       native.control[1].bc.type == ["flux", "flux", "pressure", "pressure"]
                       and np.allclose(native.control[1].bc.components[1], [0.65, 0.3, 0.05])
                       and np.allclose(native.control[1].src.components[0], [0.05, 0.9, 0.05]))
    failures += _check("forcing keys are consumed and metadata remains",
                       native.control[0].extra == {"name": "forcing-a"}
                       and native.control[1].extra == {"name": "forcing-b"})
    return failures


def validate_whole_schedule_mapping_runs() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    native = simple_schedule_from_mrst_schedule(_mrst_like_schedule(case), grid=case["grid"])
    result = run_schedule(case["overall_model"], case["initial_state"], native)

    failures = 0
    failures += _check("converted whole schedule runs all report steps",
                       len(result.states) == 3 and result.iterations.size == 3)
    failures += _check("converted whole schedule produces finite states",
                       np.isfinite(result.final_state.pressure).all()
                       and np.isfinite(result.final_state.z).all())
    failures += _check("converted whole schedule restores model forcing",
                       case["overall_model"].bc is case["bc"]
                       and getattr(case["overall_model"], "src", None) is None)
    return failures


def validate_single_control_mapping() -> int:
    case = make_boundary_conditions_case(nx=4, ny=2)
    schedule = {
        "step": {"val": [case["schedule"].step.val[0]]},
        "control": {"bc": case["bc"], "label": "single-control"},
    }
    native = simple_schedule_from_mrst_schedule(schedule, grid=case["grid"])

    failures = 0
    failures += _check("single mapping control is treated as one control block",
                       len(native.control) == 1 and native.step.control.tolist() == [1])
    failures += _check("single mapping control preserves native BC and metadata",
                       native.control[0].bc is not case["bc"]
                       and np.array_equal(native.control[0].bc.face, case["bc"].face)
                       and native.control[0].extra == {"label": "single-control"})
    return failures


def main() -> int:
    print("\nPhase 188: whole MRST-like schedule mapping conversion")
    failures = validate_whole_schedule_mapping_conversion()
    failures += validate_whole_schedule_mapping_runs()
    failures += validate_single_control_mapping()
    if failures:
        print(f"\nPhase 188 failed ({failures} checks).")
        return 1
    print("\nWhole MRST-like schedule mappings now convert to native compositional schedules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
