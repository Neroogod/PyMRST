"""Phase 332: FacilityModel setWellTargets update behavior."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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


def _well(name: str, control: str, target: float, compi, sign: int = -1) -> Well:
    return Well(
        cells=np.array([0]),
        type=control,
        val=target,
        r=np.array([0.1]),
        dir="z",
        WI=np.array([1.0e-12]),
        dZ=np.array([0.0]),
        name=name,
        compi=np.asarray(compi, dtype=float),
        refDepth=0.0,
        sign=sign,
        status=True,
    )


def _problem(names):
    return FacilityLinearizedProblem([], [], [], names, {})


def validate_target_reset_enabled() -> int:
    wells = [
        _well("I1", "rate", 10.0, [0.25, 0.75], sign=1),
        _well("P1", "bhp", 123.0, [0.0, 1.0], sign=-1),
        _well("P2", "orat", -4.0, [0.0, 1.0], sign=-1),
        _well("P3", "lrat", -8.0, [0.25, 0.75], sign=-1),
    ]
    facility = FacilityModel(wells)
    well_sol = [
        SimpleNamespace(status=True, qWs=100.0, qOs=100.0, bhp=50.0),
        SimpleNamespace(status=True, qWs=100.0, qOs=100.0, bhp=50.0),
        SimpleNamespace(status=True, qWs=100.0, qOs=100.0, bhp=50.0),
        SimpleNamespace(status=True, qWs=100.0, qOs=100.0, bhp=50.0),
    ]
    dx = {
        "qWs": np.array([1.0, 1.0, 1.0, 1.0]),
        "qOs": np.array([2.0, 2.0, 2.0, 2.0]),
        "bhp": np.array([3.0, 3.0, 3.0, 3.0]),
    }
    updated, rest = facility.updateWellSol(well_sol, _problem(["qWs", "qOs", "bhp"]), dx)

    failures = 0
    failures += _check("setWellTargets keeps updateWellSol return shape",
                       updated is well_sol and rest == [])
    failures += _check("total rate controls reset phase rates from compi",
                       np.isclose(well_sol[0].qWs, 2.5)
                       and np.isclose(well_sol[0].qOs, 7.5))
    failures += _check("bhp controls reset wellSol.bhp to the scheduled target",
                       np.isclose(well_sol[1].bhp, 123.0))
    failures += _check("phase-specific rate controls reset the controlled phase",
                       np.isclose(well_sol[2].qOs, -4.0))
    failures += _check("liquid-rate controls split target over water/oil compi",
                       np.isclose(well_sol[3].qWs, -2.0)
                       and np.isclose(well_sol[3].qOs, -6.0))
    return failures


def validate_target_reset_disabled() -> int:
    facility = FacilityModel([_well("I1", "rate", 10.0, [0.25, 0.75], sign=1)])
    facility.setWellTargets = False
    well_sol = [SimpleNamespace(status=True, qWs=100.0, qOs=100.0, bhp=50.0)]
    dx = {
        "qWs": np.array([1.0]),
        "qOs": np.array([2.0]),
        "bhp": np.array([3.0]),
    }
    facility.updateWellSol(well_sol, _problem(["qWs", "qOs", "bhp"]), dx)

    failures = 0
    failures += _check("setWellTargets false preserves Newton-updated values",
                       np.isclose(well_sol[0].qWs, 101.0)
                       and np.isclose(well_sol[0].qOs, 102.0)
                       and np.isclose(well_sol[0].bhp, 53.0))
    return failures


def main() -> int:
    print("\nPhase 332: FacilityModel setWellTargets update behavior")
    failures = 0
    failures += validate_target_reset_enabled()
    failures += validate_target_reset_disabled()
    if failures:
        print(f"\nPhase 332 failed ({failures} checks).")
        return 1
    print("\nFacilityModel setWellTargets behavior is validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
