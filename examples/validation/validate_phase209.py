"""Phase 209: packed fallback reports recover schedule timestep/control metadata."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    ScheduleControl,
    SimulationResult,
    get_multiple_packed_simulator_outputs,
    pack_simulation_problem,
    simple_schedule,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def validate_schedule_metadata_recovery() -> int:
    schedule = simple_schedule(
        [1.0, 2.0, 4.0],
        control=[ScheduleControl(), ScheduleControl()],
        step_control=[1, 2, 2],
    )
    problem = pack_simulation_problem(
        0.0,
        object(),
        schedule,
        "ValidationSuite",
        name="ScheduleAwareLegacyProblem",
    )
    problem.result = SimulationResult(
        states=[1.0, 2.0, 3.0],
        iterations=np.asarray([7, 8, 9], dtype=int),
        reports=None,
    )
    _, _, reports, _ = get_multiple_packed_simulator_outputs([problem])
    fallback = reports[0]

    failures = 0
    failures += _check("fallback recovers schedule timesteps",
                       np.allclose([r["Timestep"] for r in fallback], [1.0, 2.0, 4.0]))
    failures += _check("fallback recovers mini-step singleton lists",
                       [r["MiniSteps"] for r in fallback] == [[1.0], [2.0], [4.0]])
    failures += _check("fallback recovers one-based controls",
                       [r["Control"] for r in fallback] == [1, 2, 2])
    failures += _check("fallback derives control-change flags from schedule controls",
                       [r["ControlChanged"] for r in fallback] == [True, True, False])
    failures += _check("fallback keeps iteration vector unchanged",
                       [r["Iterations"] for r in fallback] == [7, 8, 9])
    failures += _check("schedule metadata recovery remains non-facility metadata",
                       all(not r["FacilityChanged"] for r in fallback)
                       and all(not r["ForcingChanged"] for r in fallback))
    return failures


def validate_mismatched_legacy_lengths_stay_conservative() -> int:
    schedule = simple_schedule([1.0], step_control=[1])
    problem = pack_simulation_problem(
        0.0,
        object(),
        schedule,
        "ValidationSuite",
        name="LongLegacyProblem",
    )
    problem.result = SimulationResult(
        states=[1.0, 2.0],
        iterations=np.asarray([3, 4], dtype=int),
        reports=None,
    )
    _, _, reports, _ = get_multiple_packed_simulator_outputs([problem])
    fallback = reports[0]

    failures = 0
    failures += _check("first mismatched fallback row uses available timestep",
                       np.isclose(fallback[0]["Timestep"], 1.0)
                       and fallback[0]["MiniSteps"] == [1.0])
    failures += _check("extra mismatched fallback rows keep unknown timestep",
                       np.isnan(fallback[1]["Timestep"])
                       and fallback[1]["MiniSteps"] == [])
    failures += _check("extra mismatched fallback rows use default control",
                       fallback[1]["Control"] == 1)
    return failures


def main() -> int:
    print("\nPhase 209: packed fallback schedule metadata")
    failures = validate_schedule_metadata_recovery()
    failures += validate_mismatched_legacy_lengths_stay_conservative()
    if failures:
        print(f"\nPhase 209 failed ({failures} checks).")
        return 1
    print("\nPacked legacy reports now recover schedule timestep/control metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
