"""Phase 322: standalone FacilityModel.getEquations parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities import FacilityLinearizedProblem, FacilityModel  # noqa: E402
from pymrst.params.wells_and_bc import Well  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


class TinyReservoirModel:
    n_phases = 2
    n_components = 2
    gravity_magnitude = 0.0

    def get_props(self, state, *names):
        out = []
        for name in names:
            if name == "Mobility":
                out.append([np.ones(2), np.ones(2)])
            elif name == "ShrinkageFactors":
                out.append([np.ones(2), np.ones(2)])
            elif name == "Density":
                out.append([np.full(2, 1000.0), np.full(2, 800.0)])
            else:
                raise KeyError(name)
        return out[0] if len(out) == 1 else out

    def getPhaseNames(self):
        return ["W", "O"]

    def getNumberOfComponents(self):
        return 2

    def getDissolutionMatrix(self, rs, rv):
        return None


def _well() -> Well:
    return Well(
        cells=np.array([0, 1], dtype=np.int64),
        type="rate",
        val=1.0,
        r=np.array([0.1, 0.1]),
        dir="z",
        WI=np.array([1.0, 2.0]),
        dZ=np.array([0.0, 0.0]),
        name="P1",
        compi=np.array([0.5, 0.5]),
        refDepth=0.0,
        sign=-1,
        status=True,
    )


def validate_facility_get_equations() -> int:
    facility = FacilityModel([_well()])
    facility.setReservoirModel(TinyReservoirModel())
    state0 = {
        "pressure": np.array([100.0, 95.0]),
        "wellSol": [{"status": True, "qWs": 0.2, "qOs": 0.8, "bhp": 80.0}],
    }
    state = {
        "pressure": np.array([101.0, 96.0]),
        "wellSol": [{"status": True, "qWs": 0.25, "qOs": 0.75, "bhp": 82.0}],
        "components": np.column_stack([np.ones(2), np.zeros(2)]),
    }

    problem, out_state = facility.getEquations(
        state0,
        state,
        10.0,
        None,
        "resOnly",
        True,
        "iteration",
        4,
    )
    assembled = problem.assembleSystem()
    convergence, values, names, evaluated = facility.checkFacilityConvergence(problem)

    failures = 0
    failures += _check("getEquations returns a FacilityLinearizedProblem and same state object",
                       isinstance(problem, FacilityLinearizedProblem)
                       and out_state is state)
    failures += _check("problem contains base well equations plus control equation",
                       problem.equationNames == ["wellW", "wellO", "control"]
                       and problem.types == ["perf", "perf", "well"]
                       and len(problem.equations) == 3)
    failures += _check("primary variables follow facility getAllPrimaryVariables order",
                       problem.primaryVariables == ["qWs", "qOs", "bhp"])
    failures += _check("problem records MRST-style options and source/well systems",
                       problem.iterationNo == 4
                       and problem.metadata["res_only"]
                       and problem.sources.sourceCells.tolist() == [0, 1]
                       and problem.wellSystem.controlEquation is problem.equations[-1])
    failures += _check("assembleSystem creates finite residual vector",
                       assembled is problem
                       and problem.assembled
                       and problem.b is not None
                       and np.all(np.isfinite(problem.b))
                       and problem.b.size > 0)
    failures += _check("indexOfType and convergence helpers see facility equation types",
                       np.array_equal(problem.indexOfType("perf"), np.array([True, True, False]))
                       and evaluated.tolist() == [True, True, True]
                       and len(convergence) == len(values) == len(names) == 3)
    return failures


def main() -> int:
    print("\nPhase 322: standalone FacilityModel.getEquations parity")
    failures = validate_facility_get_equations()
    if failures:
        print(f"\nPhase 322 failed ({failures} checks).")
        return 1
    print("\nFacilityModel.getEquations standalone problem assembly is validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
