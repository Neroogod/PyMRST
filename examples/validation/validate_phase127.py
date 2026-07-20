"""Phase 127: schedule-level compositional facility control blocks.

MRST schedules carry force/facility controls as report-step control blocks. This
native PyMRST regression checks that ``simple_schedule`` can carry source,
separator, and group-control blocks directly, and that ``run_schedule`` uses
isolated copies so model-side switching cannot mutate the stored schedule.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    ScheduleControl,
    run_schedule,
    simple_schedule,
)
from pymrst.modules.compositional.model import FacilityGroupControl  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class MutatingFacilityModel:
    """Tiny schedule-runner stand-in that mutates active facility blocks."""

    def __init__(self):
        self.wells = []
        self.bc = None
        self.src = {"model": "original"}
        self.separator_group = {"model": "original"}
        self.group_controls = [
            FacilityGroupControl("MODEL", wells=[], lims={"orat": 9.0})
        ]
        self.seen = []

    def step(self, state, dt, bc=None, src=None):
        self.seen.append(
            (
                None if self.src is None else dict(self.src),
                None if self.separator_group is None else dict(self.separator_group),
                [g.name for g in self.group_controls],
                [dict(g.lims) for g in self.group_controls],
            )
        )
        if self.src is not None:
            self.src["mutated"] = True
        if self.separator_group is not None:
            self.separator_group["mutated"] = True
        if self.group_controls:
            self.group_controls[0].lims["orat"] = 123.0
        return state + 1, True, 1


def validate_simple_schedule_facility_keywords() -> int:
    src = {"name": "SRC-A"}
    separator = {"name": "SEP-A"}
    group = FacilityGroupControl(
        "GROUP-A",
        wells=["P1", "P2"],
        lims={"orat": 5.0e-5},
        allocation="fraction",
        fractions=[0.25, 0.75],
    )
    schedule = simple_schedule(
        np.array([1.0, 2.0]),
        src=src,
        separator_group=separator,
        group_controls=[group],
    )

    failures = 0
    failures += _check("one-control facility schedule uses one-based controls",
                       schedule.step.control.tolist() == [1, 1])
    failures += _check("source block retained on control",
                       schedule.control[0].src is src)
    failures += _check("separator block retained on control",
                       schedule.control[0].separator_group is separator)
    failures += _check("group-control block retained on control",
                       schedule.control[0].group_controls == [group])
    return failures


def validate_facility_block_copy_isolation() -> int:
    src = {"name": "SRC-A"}
    separator = {"name": "SEP-A"}
    group = FacilityGroupControl("GROUP-A", wells=["P1"], lims={"orat": 1.0})
    model = MutatingFacilityModel()
    schedule = simple_schedule(
        [1.0],
        src=src,
        separator_group=separator,
        group_controls=[group],
    )
    out = run_schedule(model, 0, schedule)

    failures = 0
    failures += _check("facility schedule advanced one step", out.final_state == 1)
    failures += _check("model source restored",
                       model.src == {"model": "original"})
    failures += _check("model separator restored",
                       model.separator_group == {"model": "original"})
    failures += _check("model group controls restored",
                       model.group_controls[0].name == "MODEL"
                       and model.group_controls[0].lims == {"orat": 9.0})
    failures += _check("schedule source not mutated",
                       schedule.control[0].src == {"name": "SRC-A"})
    failures += _check("schedule separator not mutated",
                       schedule.control[0].separator_group == {"name": "SEP-A"})
    failures += _check("schedule group limit not mutated",
                       schedule.control[0].group_controls[0].lims == {"orat": 1.0})
    failures += _check("active copies reached model step",
                       model.seen == [
                           (
                               {"name": "SRC-A"},
                               {"name": "SEP-A"},
                               ["GROUP-A"],
                               [{"orat": 1.0}],
                           )
                       ])
    return failures


def validate_multi_control_group_switching() -> int:
    group_a = FacilityGroupControl("GROUP-A", wells=["P1"], lims={"orat": 1.0})
    group_b = FacilityGroupControl("GROUP-B", wells=["P2"], lims={"orat": 2.0})
    model = MutatingFacilityModel()
    schedule = simple_schedule(
        [1.0, 1.0],
        control=[
            ScheduleControl(group_controls=[group_a]),
            ScheduleControl(group_controls=[group_b]),
        ],
        step_control=[1, 2],
    )
    out = run_schedule(model, 0, schedule)

    failures = 0
    failures += _check("two facility controls advanced", out.final_state == 2)
    failures += _check("group names switched by report step",
                       [seen[2][0] for seen in model.seen] == ["GROUP-A", "GROUP-B"])
    failures += _check("first schedule group preserved",
                       schedule.control[0].group_controls[0].lims == {"orat": 1.0})
    failures += _check("second schedule group preserved",
                       schedule.control[1].group_controls[0].lims == {"orat": 2.0})
    return failures


def main() -> int:
    print("\nPhase 127: schedule-level facility control blocks")
    failures = validate_simple_schedule_facility_keywords()
    failures += validate_facility_block_copy_isolation()
    failures += validate_multi_control_group_switching()
    if failures:
        print(f"\nPhase 127 failed ({failures} checks).")
        return 1
    print("\nSchedule-level compositional facility controls are isolated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
