"""Phase 316: MRST FacilityModel extra/all primary-variable parity."""

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


class _WellModel:
    def __init__(self, alpha: float, beta: float):
        self.alpha = alpha
        self.beta = beta

    def getExtraPrimaryVariables(self, well_sol, reservoir_model):
        return [np.array([self.alpha]), np.array([self.beta])], ["alpha", "beta"]


def validate_extra_and_all_primary_variables() -> int:
    facility = FacilityModel([_well("I1"), _well("P1", status=False), _well("P2")])
    facility.addedPrimaryVarNames = ["alpha", "beta"]
    facility.addedPrimaryVarNamesIsFromResModel = [False, False]
    facility.WellModels = [_WellModel(10.0, 20.0), _WellModel(90.0, 90.0), _WellModel(30.0, 40.0)]
    facility.well_models = facility.WellModels
    well_sol = [
        SimpleNamespace(status=True, qWs=1.0, qOs=2.0, qGs=3.0, bhp=100.0),
        SimpleNamespace(status=True, qWs=9.0, qOs=9.0, qGs=9.0, bhp=900.0),
        SimpleNamespace(status=True, qWs=4.0, qOs=5.0, qGs=6.0, bhp=200.0),
    ]

    extra, enames, extra_map = facility.getExtraPrimaryVariables(well_sol)
    all_vars, all_names, all_map = facility.getAllPrimaryVariables(well_sol)
    primary_vars, primary_names, origin = facility.getPrimaryVariables({"wellSol": well_sol})

    failures = 0
    failures += _check("primary variable names append added variables after basics",
                       facility.getPrimaryVariableNames() == ["qWs", "qOs", "qGs", "bhp", "alpha", "beta"])
    failures += _check("extra primary variables stack active wells in MRST order",
                       enames == ["alpha", "beta"]
                       and np.allclose(extra[0], [10.0, 30.0])
                       and np.allclose(extra[1], [20.0, 40.0]))
    failures += _check("extra primary variable map is active-well by added-variable",
                       extra_map.shape == (2, 2)
                       and np.array_equal(extra_map, [[1, 2], [1, 2]]))
    failures += _check("all primary variables concatenate basic and extra blocks",
                       all_names == ["qWs", "qOs", "qGs", "bhp", "alpha", "beta"]
                       and np.allclose(all_vars[0], [1.0, 4.0])
                       and np.allclose(all_vars[3], [100.0, 200.0])
                       and np.allclose(all_vars[4], [10.0, 30.0]))
    failures += _check("all-primary map extends rate/bhp flags and carries extraMap",
                       np.array_equal(all_map["isBHP"], [False, False, False, True, False, False])
                       and np.array_equal(all_map["isRate"], [True, True, True, False, False, False])
                       and np.array_equal(all_map["extraMap"], extra_map))
    failures += _check("getPrimaryVariables includes extra variables and facility origin",
                       primary_names == all_names
                       and all(np.allclose(a, b) for a, b in zip(primary_vars, all_vars))
                       and origin == ["FacilityModel"] * len(all_names))
    failures += _check("getWellVariableMap recognizes added variables",
                       np.array_equal(facility.getWellVariableMap("alpha", well_sol), [0, 1]))
    return failures


def main() -> int:
    print("\nPhase 316: MRST FacilityModel extra/all primary-variable parity")
    failures = validate_extra_and_all_primary_variables()
    if failures:
        print(f"\nPhase 316 failed ({failures} checks).")
        return 1
    print("\nFacilityModel extra/all primary-variable helpers are validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
