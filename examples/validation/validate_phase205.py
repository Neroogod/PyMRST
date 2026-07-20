"""Phase 205: fixed-step forcing report metadata."""

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


class ReportingModel:
    def __init__(self):
        self.seen_bc = []

    def step(self, state, dt, bc=None):
        self.seen_bc.append(bc)
        return state + dt, True, 1


def validate_run_steps_with_boundary() -> int:
    model = ReportingModel()
    result = run_steps(model, 0.0, [1.0 * DAY, 1.0 * DAY], bc={"bc": "fixed"})
    reports = result.reports

    failures = 0
    failures += _check("run_steps keeps fixed boundary condition active",
                       model.seen_bc == [{"bc": "fixed"}, {"bc": "fixed"}])
    failures += _check("run_steps reports first fixed-BC forcing change",
                       [r["BoundaryChanged"] for r in reports] == [True, False]
                       and [r["ForcingChanged"] for r in reports] == [True, False])
    failures += _check("run_steps reports no source changes",
                       [r["SourceChanged"] for r in reports] == [False, False])
    failures += _check("run_steps control-change semantics remain one-control",
                       [r["ControlChanged"] for r in reports] == [True, False])
    return failures


def validate_run_steps_without_boundary() -> int:
    result = run_steps(ReportingModel(), 0.0, [1.0 * DAY])
    report = result.reports[0]

    failures = 0
    failures += _check("run_steps without BC reports no boundary change",
                       report["BoundaryChanged"] is False)
    failures += _check("run_steps without BC reports no forcing change",
                       report["ForcingChanged"] is False)
    failures += _check("run_steps without BC reports no source change",
                       report["SourceChanged"] is False)
    return failures


def main() -> int:
    print("\nPhase 205: fixed-step forcing report metadata")
    failures = validate_run_steps_with_boundary()
    failures += validate_run_steps_without_boundary()
    if failures:
        print(f"\nPhase 205 failed ({failures} checks).")
        return 1
    print("\nFixed-step compositional reports now expose forcing-change metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
