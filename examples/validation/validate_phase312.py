"""Phase 312: MRST FacilityModel basic updateWellSol parity."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities import FacilityModel  # noqa: E402
from pymrst.params.wells_and_bc import Well  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _well(name: str, *, status: bool = True) -> Well:
    return Well(
        cells=np.array([0], dtype=np.int64),
        type="rate",
        val=1.0,
        r=np.array([0.1]),
        dir="z",
        WI=np.array([1.0]),
        dZ=np.array([0.0]),
        name=name,
        compi=np.array([0.2, 0.3, 0.5]),
        refDepth=0.0,
        sign=1,
        status=status,
    )


def validate_update_well_sol_basic_variables() -> int:
    facility = FacilityModel([_well("I1"), _well("P1", status=False), _well("P2")])
    well_sol = [
        SimpleNamespace(status=True, qWs=1.0, qOs=2.0, qGs=3.0, bhp=100.0),
        SimpleNamespace(status=True, qWs=9.0, qOs=9.0, qGs=9.0, bhp=900.0),
        SimpleNamespace(status=True, qWs=4.0, qOs=5.0, qGs=6.0, bhp=200.0),
    ]
    problem = {"primaryVariables": ["qWs", "qOs", "qGs", "bhp", "pressure"]}
    dx = {
        "qWs": np.array([0.1, 0.2]),
        "qOs": np.array([0.3, 0.4]),
        "qGs": np.array([0.5, 0.6]),
        "bhp": np.array([10.0, 20.0]),
    }
    out, rest = facility.updateWellSol(well_sol, problem, dx)

    mass_facility = FacilityModel([_well("I1"), _well("P1"), _well("P2")],
                                  primary_variable_set="bhp_massfractions")
    mass_sol = [
        {"status": True, "bhp": 100.0, "massfractions": np.array([0.2, 0.3, 0.5])},
        {"status": False, "bhp": 900.0, "massfractions": np.array([0.8, 0.1, 0.1])},
        {"status": True, "bhp": 200.0, "massfractions": np.array([0.1, 0.4, 0.5])},
    ]
    mass_problem = {"primary_variables": ["bhp", "well_massfractions", "pressure"]}
    mass_out, mass_rest = mass_facility.update_well_sol(
        mass_sol,
        mass_problem,
        [np.array([1.0, 2.0]), np.array([0.01, -0.02, 0.03, -0.04])],
    )

    failures = 0
    failures += _check("basic well variables update active wells only",
                       np.isclose(out[0].qWs, 1.1)
                       and np.isclose(out[0].qOs, 2.3)
                       and np.isclose(out[0].qGs, 3.5)
                       and np.isclose(out[0].bhp, 110.0)
                       and np.isclose(out[2].qWs, 4.2)
                       and np.isclose(out[2].qOs, 5.4)
                       and np.isclose(out[2].qGs, 6.6)
                       and np.isclose(out[2].bhp, 220.0))
    failures += _check("inactive wellSol rows are not updated",
                       np.isclose(out[1].qWs, 9.0)
                       and np.isclose(out[1].bhp, 900.0))
    failures += _check("consumed facility variables are stripped from rest_vars",
                       rest == ["pressure"])
    failures += _check("bhp_massfractions updates dict wellSol rows",
                       np.isclose(mass_out[0]["bhp"], 101.0)
                       and np.isclose(mass_out[2]["bhp"], 202.0)
                       and np.allclose(mass_out[0]["massfractions"], [0.21, 0.28, 0.51])
                       and np.allclose(mass_out[2]["massfractions"], [0.13, 0.36, 0.51]))
    failures += _check("bhp_massfractions leaves inactive dict row and rest vars",
                       np.isclose(mass_out[1]["bhp"], 900.0)
                       and np.allclose(mass_out[1]["massfractions"], [0.8, 0.1, 0.1])
                       and mass_rest == ["pressure"])
    return failures


def main() -> int:
    print("\nPhase 312: MRST FacilityModel basic updateWellSol parity")
    failures = validate_update_well_sol_basic_variables()
    if failures:
        print(f"\nPhase 312 failed ({failures} checks).")
        return 1
    print("\nFacilityModel basic wellSol updates are validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
