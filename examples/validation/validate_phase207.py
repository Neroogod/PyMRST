"""Phase 207: fixed-step report schema parity."""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import run_steps  # noqa: E402
from pymrst.utils.units import DAY  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class StepModel:
    def step(self, state, dt, bc=None):
        return state + dt, True, 1


SCHEDULE_REPORT_KEYS = {
    "FacilityChanged",
    "WellControlsChanged",
    "BoundaryChanged",
    "SourceChanged",
    "ForcingChanged",
    "SeparatorChanged",
    "GroupControlsChanged",
    "VFPInjectorTablesChanged",
    "VFPProducerTablesChanged",
    "VFPTablesChanged",
    "WellSwitches",
    "GroupSwitches",
    "EndRunRequested",
    "EndRunGroups",
    "SeparatorAllocationDiagnostics",
    "SeparatorAllocationSummary",
}


def validate_fixed_step_schema_defaults() -> int:
    result = run_steps(StepModel(), 0.0, [1.0 * DAY, 1.0 * DAY])
    reports = result.reports

    failures = 0
    failures += _check("fixed-step reports expose schedule report keys",
                       all(SCHEDULE_REPORT_KEYS <= set(r) for r in reports))
    failures += _check("fixed-step facility flags default false",
                       all(
                           r["FacilityChanged"] is False
                           and r["WellControlsChanged"] is False
                           and r["SeparatorChanged"] is False
                           and r["GroupControlsChanged"] is False
                           and r["VFPTablesChanged"] is False
                           and r["VFPInjectorTablesChanged"] is False
                           and r["VFPProducerTablesChanged"] is False
                           for r in reports
                       ))
    failures += _check("fixed-step switch diagnostics default empty",
                       all(r["WellSwitches"] == [] and r["GroupSwitches"] == [] for r in reports))
    failures += _check("fixed-step end-run diagnostics default false/empty",
                       all(r["EndRunRequested"] is False and r["EndRunGroups"] == [] for r in reports))
    failures += _check("fixed-step separator allocation diagnostics default empty",
                       all(
                           r["SeparatorAllocationDiagnostics"] is None
                           and r["SeparatorAllocationSummary"] is None
                           for r in reports
                       ))
    return failures


def validate_fixed_step_schema_with_forcing() -> int:
    result = run_steps(StepModel(), 0.0, [1.0 * DAY], bc={"bc": "fixed"})
    report = result.reports[0]

    failures = 0
    failures += _check("fixed-step forcing keeps facility defaults false",
                       report["BoundaryChanged"] is True
                       and report["ForcingChanged"] is True
                       and report["FacilityChanged"] is False
                       and report["WellSwitches"] == []
                       and report["GroupSwitches"] == [])
    return failures


def main() -> int:
    print("\nPhase 207: fixed-step report schema parity")
    failures = validate_fixed_step_schema_defaults()
    failures += validate_fixed_step_schema_with_forcing()
    if failures:
        print(f"\nPhase 207 failed ({failures} checks).")
        return 1
    print("\nFixed-step reports now expose the schedule-report schema defaults.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
