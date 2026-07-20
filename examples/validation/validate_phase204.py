"""Phase 204: schedule boundary/source forcing-change diagnostics."""

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
        self.bc = {"default": "bc"}
        self.src = {"default": "src"}
        self.last_well_switches = []
        self.last_group_switches = []
        self.seen = []

    def step(self, state, dt, bc=None, src=None):
        self.last_well_switches = []
        self.last_group_switches = []
        self.seen.append((self.bc, self.src, bc, src))
        return state + dt, True, 1


def validate_managed_forcing_blocks() -> int:
    model = ReportingModel()
    schedule = simple_schedule(
        [1.0 * DAY, 1.0 * DAY, 1.0 * DAY],
        control=[
            ScheduleControl(bc={"bc": "A"}, src={"src": "A"}),
            ScheduleControl(bc={"bc": "B"}, src={"src": "B"}),
        ],
    )
    schedule.step.control[:] = [1, 1, 2]
    result = run_schedule(model, 0.0, schedule)
    reports = result.reports

    failures = 0
    failures += _check("boundary-change flag follows one-based controls",
                       [r["BoundaryChanged"] for r in reports] == [True, False, True])
    failures += _check("source-change flag follows one-based controls",
                       [r["SourceChanged"] for r in reports] == [True, False, True])
    failures += _check("combined forcing flag follows one-based controls",
                       [r["ForcingChanged"] for r in reports] == [True, False, True])
    failures += _check("forcing changes do not masquerade as facility changes",
                       [r["FacilityChanged"] for r in reports] == [False, False, False])
    failures += _check("active boundary/source blocks reach model step",
                       model.seen[0] == ({"bc": "A"}, {"src": "A"}, {"bc": "A"}, {"src": "A"})
                       and model.seen[2] == ({"bc": "B"}, {"src": "B"}, {"bc": "B"}, {"src": "B"}))
    failures += _check("model boundary/source defaults are restored after run",
                       model.bc == {"default": "bc"} and model.src == {"default": "src"})
    return failures


def validate_partial_and_unmanaged_forcing_blocks() -> int:
    failures = 0

    bc_model = ReportingModel()
    bc_schedule = simple_schedule(
        [1.0 * DAY, 1.0 * DAY],
        control=[ScheduleControl(bc={"bc": "A"}), ScheduleControl(bc={"bc": "B"})],
    )
    bc_schedule.step.control[:] = [1, 2]
    bc_reports = run_schedule(bc_model, 0.0, bc_schedule).reports
    failures += _check("boundary-only schedules do not report source changes",
                       [r["BoundaryChanged"] for r in bc_reports] == [True, True]
                       and [r["SourceChanged"] for r in bc_reports] == [False, False]
                       and [r["ForcingChanged"] for r in bc_reports] == [True, True])

    model = ReportingModel()
    schedule = simple_schedule([1.0 * DAY], control=[ScheduleControl()])
    report = run_schedule(model, 0.0, schedule).reports[0]
    failures += _check("unmanaged forcing flags stay false",
                       report["BoundaryChanged"] is False
                       and report["SourceChanged"] is False
                       and report["ForcingChanged"] is False)
    failures += _check("unmanaged forcing change stays separate from facility change",
                       report["FacilityChanged"] is False)
    failures += _check("unmanaged model boundary/source defaults are restored",
                       model.bc == {"default": "bc"} and model.src == {"default": "src"})
    return failures


def main() -> int:
    print("\nPhase 204: schedule boundary/source forcing-change diagnostics")
    failures = validate_managed_forcing_blocks()
    failures += validate_partial_and_unmanaged_forcing_blocks()
    if failures:
        print(f"\nPhase 204 failed ({failures} checks).")
        return 1
    print("\nSchedule reports now expose boundary/source forcing-block changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
