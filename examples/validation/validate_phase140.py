"""Phase 140: MRST-style compositional schedule report metadata."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    ScheduleControl,
    get_multiple_packed_simulator_outputs,
    pack_simulation_problem,
    run_schedule,
    run_steps,
    simple_schedule,
    simulate_packed_problem,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class DummyScheduleModel:
    """Small model that exercises compositional schedule mechanics."""

    def __init__(self):
        self.wells = []
        self.bc = None
        self.src = None
        self.separator_group = None
        self.group_controls = []
        self.calls: list[tuple[float, object, object]] = []

    def step(self, state, dt, bc=None, src=None):
        self.calls.append((float(dt), bc, src))
        if dt > 0.5:
            return state, False, 1
        return state + dt, True, 2


def validate_run_schedule_reports() -> int:
    source_a = {"name": "src-a"}
    source_b = {"name": "src-b"}
    schedule = simple_schedule(
        [1.0, 0.25, 0.25],
        control=[
            ScheduleControl(src=source_a),
            ScheduleControl(src=source_b),
        ],
        step_control=[1, 1, 2],
    )
    model = DummyScheduleModel()
    result = run_schedule(model, 0.0, schedule, max_cuts=3)
    reports = result.reports or []

    failures = 0
    failures += _check("reports are emitted for every report step", len(reports) == 3)
    failures += _check("iterations array remains the legacy summary",
                       result.iterations.tolist() == [5, 2, 2])
    failures += _check("cut report stores accepted ministeps",
                       reports[0]["Cuts"] == 1
                       and np.allclose(reports[0]["MiniSteps"], [0.5, 0.5])
                       and np.isclose(sum(reports[0]["MiniSteps"]), reports[0]["Timestep"]))
    failures += _check("report controls remain one-based",
                       [r["Control"] for r in reports] == [1, 1, 2])
    failures += _check("control-change flags follow MRST report-step semantics",
                       [r["ControlChanged"] for r in reports] == [True, False, True])
    failures += _check("all accepted reports are marked converged",
                       all(r["Converged"] for r in reports))
    failures += _check("final state accumulates accepted substeps only",
                       np.isclose(result.final_state, 1.5))
    failures += _check("source controls are deep-copied into model calls",
                       model.calls[0][2] == source_a
                       and model.calls[-1][2] == source_b
                       and model.src is None)
    return failures


def validate_run_steps_reports() -> int:
    class StepModel:
        def step(self, state, dt, bc=None):
            return state + dt, True, 3

    result = run_steps(StepModel(), 0.0, [0.1, 0.2])
    reports = result.reports or []
    failures = 0
    failures += _check("run_steps also returns report metadata", len(reports) == 2)
    failures += _check("run_steps reports single accepted ministep per step",
                       reports[0]["MiniSteps"] == [0.1]
                       and reports[1]["MiniSteps"] == [0.2])
    failures += _check("run_steps report iteration counts match legacy vector",
                       [r["Iterations"] for r in reports] == result.iterations.tolist())
    return failures


def validate_packed_reports_passthrough() -> int:
    model = DummyScheduleModel()
    schedule = simple_schedule([1.0])
    problem = pack_simulation_problem(
        0.0,
        model,
        schedule,
        "ValidationSuite",
        name="ScheduleReportProblem",
    )
    simulate_packed_problem(problem, max_cuts=3)
    _, states, reports, names = get_multiple_packed_simulator_outputs([problem])

    failures = 0
    failures += _check("packed helper preserves problem name",
                       names == ["ScheduleReportProblem"])
    failures += _check("packed helper returns state histories",
                       len(states[0]) == 1 and np.isclose(states[0][-1], 1.0))
    failures += _check("packed helper passes through rich reports",
                       reports[0][0]["Cuts"] == 1
                       and reports[0][0]["MiniSteps"] == [0.5, 0.5]
                       and reports[0][0]["Control"] == 1)
    return failures


def main() -> int:
    print("\nPhase 140: MRST-style compositional schedule report metadata")
    failures = validate_run_schedule_reports()
    failures += validate_run_steps_reports()
    failures += validate_packed_reports_passthrough()
    if failures:
        print(f"\nPhase 140 failed ({failures} checks).")
        return 1
    print("\nCompositional schedule reports expose controls, cuts, and ministeps.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
