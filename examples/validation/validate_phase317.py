"""Phase 317: MRST FacilityModel initStateAD/validateState/updateState parity."""

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


def _well(name: str, *, typ: str = "rate", val: float = 3.0) -> Well:
    return Well(
        cells=np.array([0, 1], dtype=np.int64),
        type=typ,
        val=val,
        r=np.array([0.1, 0.1]),
        dir="z",
        WI=np.array([1.0, 1.0]),
        dZ=np.array([0.0, 0.0]),
        name=name,
        compi=np.array([0.2, 0.3, 0.5]),
        refDepth=0.0,
        sign=1,
        status=True,
    )


class _Reservoir:
    def getPhaseNames(self):
        return ("W", "O", "G")

    def getNumberOfComponents(self):
        return 3


def validate_state_hooks() -> int:
    facility = FacilityModel([_well("I1"), _well("P1", typ="bhp", val=120.0)],
                             primary_variable_set="bhp_massfractions")
    facility.ReservoirModel = _Reservoir()
    state = {"pressure": np.array([100.0, 101.0])}
    validated = facility.validateState(state)
    well_sol = validated["wellSol"]

    variables = [
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([5.0, 6.0]),
        np.array([100.0, 200.0]),
        np.array([0.1, 0.2, 0.3, 0.4]),
    ]
    names = ["qWs", "qOs", "qGs", "bhp", "well_massfractions"]
    initialized = facility.initStateAD(validated, variables, names, ["FacilityModel"] * len(names))
    fs = initialized["FacilityState"]

    update_state = {
        "wellSol": [
            SimpleNamespace(status=True, bhp=10.0, qWs=1.0, qOs=2.0, qGs=3.0),
            SimpleNamespace(status=True, bhp=20.0, qWs=4.0, qOs=5.0, qGs=6.0),
        ]
    }
    updated, report = facility.updateState(
        update_state,
        {"primaryVariables": ["bhp", "pressure"]},
        {"bhp": np.array([1.0, 2.0])},
    )

    failures = 0
    failures += _check("validateState creates MRST-like wellSol rows",
                       len(well_sol) == 2
                       and well_sol[0].name == "I1"
                       and well_sol[0].status is True
                       and np.allclose(well_sol[0].flux, [0.0, 0.0])
                       and hasattr(well_sol[0], "qWs")
                       and np.allclose(well_sol[0].massfractions, [1/3, 1/3, 1/3]))
    failures += _check("initStateAD stores primary variables without well_massfractions",
                       fs["names"] == ["qWs", "qOs", "qGs", "bhp"]
                       and len(fs["primaryVariables"]) == 4
                       and all(np.allclose(a, b) for a, b in zip(fs["surfacePhaseRates"], variables[:3])))
    failures += _check("initStateAD expands well_massfractions with final fill component",
                       np.allclose(fs["massfractions_matrix"], [[0.1, 0.2, 0.7], [0.3, 0.4, 0.3]]))
    failures += _check("initStateAD writes snake and MRST FacilityState aliases",
                       initialized["FacilityState"] is initialized["facility_state"])
    failures += _check("updateState delegates to updateWellSol and returns empty report",
                       updated is update_state
                       and report == []
                       and np.allclose([updated["wellSol"][0].bhp, updated["wellSol"][1].bhp], [11.0, 22.0]))
    return failures


def main() -> int:
    print("\nPhase 317: MRST FacilityModel initStateAD/validateState/updateState parity")
    failures = validate_state_hooks()
    if failures:
        print(f"\nPhase 317 failed ({failures} checks).")
        return 1
    print("\nFacilityModel state initialization and update hooks are validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
