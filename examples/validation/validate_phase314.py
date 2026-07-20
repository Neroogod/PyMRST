"""Phase 314: MRST FacilityModel getProp/getVariableField parity."""

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


def _well(name: str, *, status: bool = True, ncomp: int = 3) -> Well:
    return Well(
        cells=np.array([0], dtype=np.int64),
        type="rate",
        val=1.0,
        r=np.array([0.1]),
        dir="z",
        WI=np.array([1.0]),
        dZ=np.array([0.0]),
        name=name,
        compi=np.ones(ncomp) / ncomp,
        refDepth=0.0,
        sign=1,
        status=status,
    )


class _Reservoir:
    def getPhaseNames(self):
        return ("W", "O", "G")


def validate_facility_get_prop_surface() -> int:
    facility = FacilityModel([_well("I1"), _well("P1", status=False), _well("P2")])
    facility.ReservoirModel = _Reservoir()
    state = {
        "FacilityState": {
            "names": ["bhp", "customRate"],
            "primaryVariables": [np.array([101.0, 202.0]), np.array([7.0])],
        },
        "wellSol": [
            SimpleNamespace(status=True, bhp=100.0, qWs=1.0, qOs=2.0, qGs=3.0),
            SimpleNamespace(status=True, bhp=999.0, qWs=9.0, qOs=9.0, qGs=9.0),
            {"status": True, "bhp": 200.0, "qWs": 4.0, "qOs": 5.0, "qGs": 6.0},
        ],
        "plain": np.array([42.0]),
    }
    no_primary_state = dict(state)
    no_primary_state.pop("FacilityState")

    bhp_field, bhp_index = facility.getVariableField("bhp")
    q_field, q_index = facility.get_variable_field("qOs")
    surface_rates = facility.getProp(no_primary_state, "surfaceRates")
    q_s = facility.get_prop(no_primary_state, "q_s")
    q_w, q_o = facility.getProps(no_primary_state, "qWs", "qOs")

    failures = 0
    failures += _check("getVariableField recognizes bhp and phase rates",
                       bhp_field == "bhp"
                       and isinstance(bhp_index, slice)
                       and q_field == "qOs"
                       and isinstance(q_index, slice))
    failures += _check("getProp prioritizes FacilityState primary variables",
                       np.allclose(facility.getProp(state, "bhp"), [101.0, 202.0])
                       and np.allclose(facility.get_prop(state, "customRate"), [7.0]))
    failures += _check("getProp falls back to active wellSol fields",
                       np.allclose(facility.getProp(no_primary_state, "bhp"), [100.0, 200.0])
                       and np.allclose(q_o, [2.0, 5.0]))
    failures += _check("surfaceRates and q_s bundle phase rates in phase order",
                       len(surface_rates) == 3
                       and len(q_s) == 3
                       and np.allclose(surface_rates[0], [1.0, 4.0])
                       and np.allclose(surface_rates[1], [2.0, 5.0])
                       and np.allclose(surface_rates[2], [3.0, 6.0])
                       and all(np.allclose(a, b) for a, b in zip(surface_rates, q_s)))
    failures += _check("getProps returns multiple values in request order",
                       np.allclose(q_w, [1.0, 4.0])
                       and np.allclose(q_o, [2.0, 5.0]))
    failures += _check("unknown names can fall back to direct state fields",
                       np.allclose(facility.getProp(no_primary_state, "plain"), [42.0]))
    try:
        facility.getVariableField("notAWellVariable")
    except KeyError:
        unknown_fails = True
    else:
        unknown_fails = False
    failures += _check("strict getVariableField fails for unknown facility variables", unknown_fails)
    return failures


def main() -> int:
    print("\nPhase 314: MRST FacilityModel getProp/getVariableField parity")
    failures = validate_facility_get_prop_surface()
    if failures:
        print(f"\nPhase 314 failed ({failures} checks).")
        return 1
    print("\nFacilityModel getProp/getVariableField helpers are validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
