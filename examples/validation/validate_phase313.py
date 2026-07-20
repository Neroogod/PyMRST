"""Phase 313: MRST FacilityModel after-step wellSol parity."""

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


def _well(
    name: str,
    *,
    typ: str = "rate",
    val: float = 1.0,
    status: bool = True,
    sign: int = 1,
) -> Well:
    return Well(
        cells=np.array([0, 1], dtype=np.int64),
        type=typ,
        val=val,
        r=np.array([0.1, 0.1]),
        dir="z",
        WI=np.array([1.0, 2.0]),
        dZ=np.array([0.0, 1.0]),
        name=name,
        compi=np.array([0.2, 0.3, 0.5]),
        refDepth=10.0,
        sign=sign,
        status=status,
        cstatus=np.array([True, False]),
        components=np.array([0.1, 0.2, 0.7]),
        thp=12.0,
        vfp_index=None,
    )


def validate_update_well_sol_after_step() -> int:
    facility = FacilityModel([
        _well("I1", typ="bhp", val=250.0),
        _well("P1", typ="orat", val=-5.0, status=False, sign=-1),
    ])
    well_sol = [
        SimpleNamespace(status=True, qWs=1.2, bhp=101.0, type="rate", val=1.0),
        {"status": True, "qOs": -2.5, "bhp": 90.0, "type": "rate", "val": -1.0},
    ]
    out = facility.updateWellSolAfterStep(well_sol, well_sol)

    state = {"wellSol": [
        SimpleNamespace(status=False, qWs=3.0, bhp=105.0),
        {"status": True, "qOs": -1.0, "bhp": 80.0},
    ]}
    updated, report = facility.updateAfterConvergence({"wellSol": well_sol}, state, 1.0, None)

    obj_state = SimpleNamespace(wellSol=[
        SimpleNamespace(status=True, qWs=4.0, bhp=106.0),
        SimpleNamespace(status=True, qOs=-4.0, bhp=70.0),
    ])
    obj_updated, obj_report = facility.update_after_convergence(None, obj_state, 1.0)

    failures = 0
    failures += _check("after-step refreshes object well control metadata",
                       out[0].name == "I1"
                       and out[0].type == "bhp"
                       and np.isclose(out[0].val, 250.0)
                       and out[0].status is True
                       and np.allclose(out[0].compi, [0.2, 0.3, 0.5])
                       and np.array_equal(out[0].cstatus, [True, False])
                       and np.isclose(out[0].qWs, 1.2))
    failures += _check("after-step refreshes dict well control metadata",
                       out[1]["name"] == "P1"
                       and out[1]["type"] == "orat"
                       and np.isclose(out[1]["val"], -5.0)
                       and out[1]["status"] is False
                       and np.array_equal(out[1]["cells"], [0, 1])
                       and np.allclose(out[1]["WI"], [1.0, 2.0])
                       and np.isclose(out[1]["qOs"], -2.5))
    failures += _check("schedule arrays are copied, not aliased",
                       out[0].compi is not facility.wells[0].compi
                       and out[1]["components"] is not facility.wells[1].components)
    failures += _check("updateAfterConvergence returns state and empty report",
                       updated is state and report == [])
    failures += _check("updateAfterConvergence preserves a converged closed wellSol row",
                       updated["wellSol"][0].status is False
                       and updated["wellSol"][1]["status"] is False)
    failures += _check("object state update_after_convergence works",
                       obj_updated is obj_state
                       and obj_report == []
                       and obj_updated.wellSol[0].type == "bhp"
                       and obj_updated.wellSol[1].status is False)
    return failures


def main() -> int:
    print("\nPhase 313: MRST FacilityModel after-step wellSol parity")
    failures = validate_update_well_sol_after_step()
    if failures:
        print(f"\nPhase 313 failed ({failures} checks).")
        return 1
    print("\nFacilityModel after-step wellSol updates are validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
