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
    # `setWellTargets` is applied in `updateState`, not `updateWellSol` - MRST
    # calls `assignWellValuesFromControl` in exactly one place, and it is there,
    # driven by `drivingForces.W`. `updateWellSol` returns the raw Newton update
    # untouched, which is what phases 312/313/317 rely on. So this exercises the
    # target reset through `updateState`, with the wells supplied as driving
    # forces, which is the MRST-faithful path.
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
    state, report = facility.updateState(
        {"wellSol": well_sol}, _problem(["qWs", "qOs", "bhp"]), dx,
        {"W": wells},
    )

    failures = 0
    failures += _check("updateState returns state and empty report",
                       state["wellSol"] is well_sol and report == [])
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
    # With `setWellTargets` off, `updateState` must leave the Newton update in
    # place - and `updateWellSol`, which never assigns targets, must do so
    # regardless of the flag.
    wells = [_well("I1", "rate", 10.0, [0.25, 0.75], sign=1)]
    facility = FacilityModel(wells)
    facility.setWellTargets = False
    well_sol = [SimpleNamespace(status=True, qWs=100.0, qOs=100.0, bhp=50.0)]
    dx = {
        "qWs": np.array([1.0]),
        "qOs": np.array([2.0]),
        "bhp": np.array([3.0]),
    }
    facility.updateState({"wellSol": well_sol}, _problem(["qWs", "qOs", "bhp"]),
                         dx, {"W": wells})

    failures = 0
    failures += _check("setWellTargets false preserves Newton-updated values",
                       np.isclose(well_sol[0].qWs, 101.0)
                       and np.isclose(well_sol[0].qOs, 102.0)
                       and np.isclose(well_sol[0].bhp, 53.0))

    # and updateWellSol never applies targets, whatever the flag says
    facility.setWellTargets = True
    ws2 = [SimpleNamespace(status=True, qWs=100.0, qOs=100.0, bhp=50.0)]
    facility.updateWellSol(ws2, _problem(["qWs", "qOs", "bhp"]), dx)
    failures += _check("updateWellSol returns the raw Newton update, targets aside",
                       np.isclose(ws2[0].qWs, 101.0)
                       and np.isclose(ws2[0].bhp, 53.0))
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
