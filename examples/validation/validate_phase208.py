"""Phase 208: packed-output fallback reports use the full MRST-like schema."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.examples._shared import (  # noqa: E402
    SimulationResult,
    get_multiple_packed_simulator_outputs,
    pack_simulation_problem,
    simple_schedule,
)


EXPECTED_KEYS = {
    "Step",
    "Control",
    "ControlChanged",
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
    "Timestep",
    "Iterations",
    "Cuts",
    "MiniSteps",
    "Converged",
}


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def validate_legacy_packed_fallback_schema() -> int:
    schedule = simple_schedule([])
    problem = pack_simulation_problem(
        0.0,
        object(),
        schedule,
        "ValidationSuite",
        name="LegacyReportProblem",
    )
    problem.result = SimulationResult(
        states=[1.0, 2.0, 3.0],
        iterations=np.asarray([4, 5, 6], dtype=int),
        reports=None,
    )
    _, states, reports, names = get_multiple_packed_simulator_outputs([problem])
    fallback = reports[0]

    failures = 0
    failures += _check("packed helper preserves legacy problem name",
                       names == ["LegacyReportProblem"])
    failures += _check("packed helper preserves legacy state history",
                       states[0] == [1.0, 2.0, 3.0])
    failures += _check("fallback emits one report per legacy iteration",
                       len(fallback) == 3)
    failures += _check("fallback report uses full schedule schema",
                       set(fallback[0]) == EXPECTED_KEYS)
    failures += _check("fallback keeps legacy iteration counts",
                       [r["Iterations"] for r in fallback] == [4, 5, 6])
    failures += _check("fallback reports one-based steps and default control",
                       [r["Step"] for r in fallback] == [1, 2, 3]
                       and [r["Control"] for r in fallback] == [1, 1, 1])
    failures += _check("fallback marks only first synthetic control as changed",
                       [r["ControlChanged"] for r in fallback] == [True, False, False])
    failures += _check("fallback synthetic forcing/facility flags are false",
                       all(not r["FacilityChanged"] for r in fallback)
                       and all(not r["ForcingChanged"] for r in fallback)
                       and all(not r["VFPTablesChanged"] for r in fallback))
    failures += _check("fallback switch and end-run diagnostics are empty",
                       fallback[0]["WellSwitches"] == []
                       and fallback[0]["GroupSwitches"] == []
                       and fallback[0]["EndRunGroups"] == []
                       and not fallback[0]["EndRunRequested"])
    failures += _check("fallback records unknown timestep as NaN",
                       np.isnan(fallback[0]["Timestep"])
                       and fallback[0]["MiniSteps"] == [])
    failures += _check("fallback synthetic reports are converged no-cut reports",
                       all(r["Converged"] and r["Cuts"] == 0 for r in fallback))
    return failures


def main() -> int:
    print("\nPhase 208: packed-output fallback report schema")
    failures = validate_legacy_packed_fallback_schema()
    if failures:
        print(f"\nPhase 208 failed ({failures} checks).")
        return 1
    print("\nPacked legacy outputs now synthesize full MRST-like reports.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
