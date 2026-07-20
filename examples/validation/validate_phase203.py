"""Phase 203: schedule well/VFP facility-block change diagnostics."""

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
        self.wells = [{"name": "default"}]
        self.bc = None
        self.src = None
        self.separator_group = None
        self.group_controls = []
        self.vfp_tables_injector = [{"table": "inj-default"}]
        self.vfp_tables_producer = [{"table": "prod-default"}]
        self.last_well_switches = []
        self.last_group_switches = []
        self.seen = []
        self.rebind_count = 0

    def _bind_vfp_tables(self):
        self.rebind_count += 1

    def step(self, state, dt, bc=None, src=None):
        self.last_well_switches = []
        self.last_group_switches = []
        self.seen.append((
            self.wells,
            self.vfp_tables_injector,
            self.vfp_tables_producer,
        ))
        return state + dt, True, 1


def validate_managed_well_and_vfp_blocks() -> int:
    model = ReportingModel()
    schedule = simple_schedule(
        [1.0 * DAY, 1.0 * DAY, 1.0 * DAY],
        control=[
            ScheduleControl(
                wells=[{"name": "A"}],
                vfp_tables_injector=[{"table": "inj-A"}],
                vfp_tables_producer=[{"table": "prod-A"}],
            ),
            ScheduleControl(
                wells=[{"name": "B"}],
                vfp_tables_injector=[{"table": "inj-B"}],
                vfp_tables_producer=[{"table": "prod-B"}],
            ),
        ],
    )
    schedule.step.control[:] = [1, 1, 2]
    result = run_schedule(model, 0.0, schedule)
    reports = result.reports

    failures = 0
    failures += _check("well-control change flag follows one-based controls",
                       [r["WellControlsChanged"] for r in reports] == [True, False, True])
    failures += _check("VFP injector/prod flags follow one-based controls",
                       [r["VFPInjectorTablesChanged"] for r in reports] == [True, False, True]
                       and [r["VFPProducerTablesChanged"] for r in reports] == [True, False, True])
    failures += _check("combined VFP flag follows one-based controls",
                       [r["VFPTablesChanged"] for r in reports] == [True, False, True])
    failures += _check("facility changed includes well/VFP control blocks",
                       [r["FacilityChanged"] for r in reports] == [True, False, True])
    failures += _check("managed well/VFP blocks still reach model step",
                       model.seen[0] == ([{"name": "A"}], [{"table": "inj-A"}], [{"table": "prod-A"}])
                       and model.seen[2] == ([{"name": "B"}], [{"table": "inj-B"}], [{"table": "prod-B"}]))
    failures += _check("VFP tables are rebound for managed blocks and restore",
                       model.rebind_count == 4)
    failures += _check("model well/VFP defaults are restored after run",
                       model.wells == [{"name": "default"}]
                       and model.vfp_tables_injector == [{"table": "inj-default"}]
                       and model.vfp_tables_producer == [{"table": "prod-default"}])
    return failures


def validate_unmanaged_well_and_vfp_blocks() -> int:
    model = ReportingModel()
    schedule = simple_schedule([1.0 * DAY], control=[ScheduleControl()])
    result = run_schedule(model, 0.0, schedule)
    report = result.reports[0]

    failures = 0
    failures += _check("unmanaged well-control flag stays false",
                       report["WellControlsChanged"] is False)
    failures += _check("unmanaged VFP flags stay false",
                       report["VFPTablesChanged"] is False
                       and report["VFPInjectorTablesChanged"] is False
                       and report["VFPProducerTablesChanged"] is False)
    failures += _check("unmanaged facility change stays tied to switches",
                       report["FacilityChanged"] is False)
    failures += _check("unmanaged model well/VFP defaults remain visible",
                       model.seen == [(
                           [{"name": "default"}],
                           [{"table": "inj-default"}],
                           [{"table": "prod-default"}],
                       )])
    return failures


def main() -> int:
    print("\nPhase 203: schedule well/VFP facility-block change diagnostics")
    failures = validate_managed_well_and_vfp_blocks()
    failures += validate_unmanaged_well_and_vfp_blocks()
    if failures:
        print(f"\nPhase 203 failed ({failures} checks).")
        return 1
    print("\nSchedule reports now expose well/VFP facility-block changes.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
