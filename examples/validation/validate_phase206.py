"""Phase 206: fixed-step source forcing support."""

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
        self.seen = []

    def step(self, state, dt, bc=None, src=None):
        self.seen.append((bc, src))
        return state + dt, True, 1


def validate_run_steps_with_source() -> int:
    model = ReportingModel()
    result = run_steps(model, 0.0, [1.0 * DAY, 1.0 * DAY], src={"src": "fixed"})
    reports = result.reports

    failures = 0
    failures += _check("run_steps forwards fixed source terms",
                       model.seen == [(None, {"src": "fixed"}), (None, {"src": "fixed"})])
    failures += _check("run_steps reports first fixed-source forcing change",
                       [r["SourceChanged"] for r in reports] == [True, False]
                       and [r["ForcingChanged"] for r in reports] == [True, False])
    failures += _check("source-only run_steps reports no boundary changes",
                       [r["BoundaryChanged"] for r in reports] == [False, False])
    return failures


def validate_run_steps_with_boundary_and_source() -> int:
    model = ReportingModel()
    result = run_steps(
        model,
        0.0,
        [1.0 * DAY, 1.0 * DAY],
        bc={"bc": "fixed"},
        src={"src": "fixed"},
    )
    reports = result.reports

    failures = 0
    failures += _check("run_steps forwards fixed boundary and source together",
                       model.seen == [
                           ({"bc": "fixed"}, {"src": "fixed"}),
                           ({"bc": "fixed"}, {"src": "fixed"}),
                       ])
    failures += _check("combined fixed forcing reports first-step changes",
                       [r["BoundaryChanged"] for r in reports] == [True, False]
                       and [r["SourceChanged"] for r in reports] == [True, False]
                       and [r["ForcingChanged"] for r in reports] == [True, False])
    failures += _check("combined fixed forcing keeps one-control report semantics",
                       [r["ControlChanged"] for r in reports] == [True, False])
    return failures


def main() -> int:
    print("\nPhase 206: fixed-step source forcing support")
    failures = validate_run_steps_with_source()
    failures += validate_run_steps_with_boundary_and_source()
    if failures:
        print(f"\nPhase 206 failed ({failures} checks).")
        return 1
    print("\nFixed-step compositional runs now support source forcing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
