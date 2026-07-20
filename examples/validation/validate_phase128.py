"""Phase 128: mixed schedule facility controls clear stale blocks.

This native regression covers the MRST schedule/facility distinction that a
schedule either leaves model-level facility blocks alone, or explicitly manages
them report-step by report-step. If any schedule control manages separator or
group controls, later controls with ``None`` for that field must clear the
active block instead of reusing the previous report step's facility controls.
"""

from __future__ import annotations

import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    ScheduleControl,
    run_schedule,
    simple_schedule,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class FacilityScopeModel:
    def __init__(self):
        self.wells = []
        self.bc = None
        self.src = None
        self.separator_group = {"model": "separator-default"}
        self.group_controls = [{"model": "group-default"}]
        self.seen = []

    def step(self, state, dt, bc=None, src=None):
        self.seen.append(
            (
                copy.deepcopy(self.separator_group),
                copy.deepcopy(self.group_controls),
            )
        )
        return state + 1, True, 1


def validate_unmanaged_schedule_preserves_model_facility() -> int:
    model = FacilityScopeModel()
    schedule = simple_schedule([1.0, 1.0])
    out = run_schedule(model, 0, schedule)

    expected = [
        ({"model": "separator-default"}, [{"model": "group-default"}]),
        ({"model": "separator-default"}, [{"model": "group-default"}]),
    ]
    failures = 0
    failures += _check("unmanaged schedule advanced both reports", out.final_state == 2)
    failures += _check("model separator visible during unmanaged schedule",
                       model.seen == expected)
    failures += _check("model separator restored after unmanaged schedule",
                       model.separator_group == {"model": "separator-default"})
    failures += _check("model groups restored after unmanaged schedule",
                       model.group_controls == [{"model": "group-default"}])
    return failures


def validate_mixed_schedule_clears_stale_facility() -> int:
    model = FacilityScopeModel()
    schedule = simple_schedule(
        [1.0, 1.0, 1.0],
        control=[
            ScheduleControl(
                separator_group={"schedule": "separator-A"},
                group_controls=[{"schedule": "group-A"}],
            ),
            ScheduleControl(separator_group=None, group_controls=None),
            ScheduleControl(
                separator_group={"schedule": "separator-B"},
                group_controls=[{"schedule": "group-B"}],
            ),
        ],
        step_control=[1, 2, 3],
    )
    out = run_schedule(model, 0, schedule)

    failures = 0
    failures += _check("mixed facility schedule advanced all reports", out.final_state == 3)
    failures += _check(
        "none-valued control clears stale separator/groups",
        model.seen == [
            ({"schedule": "separator-A"}, [{"schedule": "group-A"}]),
            (None, []),
            ({"schedule": "separator-B"}, [{"schedule": "group-B"}]),
        ],
    )
    failures += _check("model separator restored after mixed schedule",
                       model.separator_group == {"model": "separator-default"})
    failures += _check("model groups restored after mixed schedule",
                       model.group_controls == [{"model": "group-default"}])
    failures += _check("first schedule group not mutated",
                       schedule.control[0].group_controls == [{"schedule": "group-A"}])
    failures += _check("third schedule group not mutated",
                       schedule.control[2].group_controls == [{"schedule": "group-B"}])
    return failures


def main() -> int:
    print("\nPhase 128: mixed schedule facility control scope")
    failures = validate_unmanaged_schedule_preserves_model_facility()
    failures += validate_mixed_schedule_clears_stale_facility()
    if failures:
        print(f"\nPhase 128 failed ({failures} checks).")
        return 1
    print("\nMixed schedule facility controls no longer leak stale blocks.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
