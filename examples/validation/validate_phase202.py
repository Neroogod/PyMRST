"""Phase 202: schedule facility-block change diagnostics."""

from __future__ import annotations

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
from pymrst.utils.units import DAY  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class ReportingModel:
    def __init__(self):
        self.wells = []
        self.bc = None
        self.src = None
        self.separator_group = {"default": "separator"}
        self.group_controls = [{"default": "group"}]
        self.last_well_switches = []
        self.last_group_switches = []
        self.seen = []

    def step(self, state, dt, bc=None, src=None):
        self.last_well_switches = []
        self.last_group_switches = []
        self.seen.append((self.separator_group, self.group_controls))
        return state + dt, True, 1


def validate_managed_facility_blocks() -> int:
    model = ReportingModel()
    schedule = simple_schedule(
        [1.0 * DAY, 1.0 * DAY, 1.0 * DAY],
        control=[
            ScheduleControl(
                separator_group={"separator": "A"},
                group_controls=[{"group": "A"}],
            ),
            ScheduleControl(
                separator_group={"separator": "B"},
                group_controls=[{"group": "B"}],
            ),
        ],
    )
    schedule.step.control[:] = [1, 1, 2]
    result = run_schedule(model, 0.0, schedule)
    reports = result.reports

    failures = 0
    failures += _check("separator change flag follows one-based controls",
                       [r["SeparatorChanged"] for r in reports] == [True, False, True])
    failures += _check("group-control change flag follows one-based controls",
                       [r["GroupControlsChanged"] for r in reports] == [True, False, True])
    failures += _check("facility changed includes managed facility blocks",
                       [r["FacilityChanged"] for r in reports] == [True, False, True])
    failures += _check("active separator/group blocks still reach model step",
                       model.seen[0] == ({"separator": "A"}, [{"group": "A"}])
                       and model.seen[2] == ({"separator": "B"}, [{"group": "B"}]))
    failures += _check("model facility defaults are restored after run",
                       model.separator_group == {"default": "separator"}
                       and model.group_controls == [{"default": "group"}])
    return failures


def validate_unmanaged_facility_blocks() -> int:
    model = ReportingModel()
    schedule = simple_schedule([1.0 * DAY], control=[ScheduleControl()])
    result = run_schedule(model, 0.0, schedule)
    report = result.reports[0]

    failures = 0
    failures += _check("unmanaged separator flag stays false",
                       report["SeparatorChanged"] is False)
    failures += _check("unmanaged group-control flag stays false",
                       report["GroupControlsChanged"] is False)
    failures += _check("unmanaged facility change stays tied to switches",
                       report["FacilityChanged"] is False)
    failures += _check("unmanaged model defaults remain visible during step",
                       model.seen == [({"default": "separator"}, [{"default": "group"}])])
    return failures


def main() -> int:
    print("\nPhase 202: schedule facility-block change diagnostics")
    failures = validate_managed_facility_blocks()
    failures += validate_unmanaged_facility_blocks()
    if failures:
        print(f"\nPhase 202 failed ({failures} checks).")
        return 1
    print("\nSchedule reports now expose separator/group-control block changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
