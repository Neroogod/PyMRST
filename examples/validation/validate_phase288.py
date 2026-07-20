"""Phase 288: generic FacilityModel group-control parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities import FacilityModel  # noqa: E402
from pymrst.modules.ad_core.statefunctions.state_function import State  # noqa: E402
from pymrst.modules.compositional.model import FacilityGroupControl  # noqa: E402
from pymrst.params.wells_and_bc import Well  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _well(name: str, sign: int = -1) -> Well:
    return Well(
        cells=np.array([0]),
        type="bhp",
        val=100.0,
        r=np.ones(1),
        dir="z",
        WI=np.ones(1),
        dZ=np.zeros(1),
        name=name,
        sign=sign,
        compi=np.array([0.0, 1.0, 0.0]),
        refDepth=0.0,
        lims={},
    )


def _facility():
    wells = [_well("P1"), _well("P2"), _well("P3")]
    return FacilityModel(wells)


def _state(rates: list[float]) -> State:
    return State(
        {
            "bhp": np.array([90.0, 91.0, 92.0], dtype=float),
            "qWs": np.zeros(3),
            "qOs": np.asarray(rates, dtype=float),
            "qGs": np.zeros(3),
        }
    )


def validate_rate_group_switch() -> int:
    facility = _facility()
    facility.group_controls = [
        FacilityGroupControl("PROD", wells=["P1", "P2"], lims={"orat": 5.0e-5})
    ]
    ok = facility.update_switched_controls(_state([-4.0e-5, -6.0e-5, 0.0]),
                                           ["qWs", "qOs", "qGs"])
    report = facility.last_group_switches[0] if facility.last_group_switches else {}

    failures = 0
    failures += _check("generic FacilityModel group rate limit switches",
                       not ok
                       and report.get("group") == "PROD"
                       and report.get("mode") == "orat")
    failures += _check("generic FacilityModel rate allocation drives targets",
                       np.isclose(facility.wells[0].val, -2.0e-5)
                       and np.isclose(facility.wells[1].val, -3.0e-5)
                       and facility.wells[0].type == facility.wells[1].type == "orat")
    failures += _check("generic FacilityModel records allocation diagnostics",
                       report.get("wells") == ["P1", "P2"]
                       and np.allclose(report.get("allocation_weights"), [0.4, 0.6]))
    return failures


def validate_fraction_and_equal_groups() -> int:
    facility = _facility()
    facility.group_controls = [
        FacilityGroupControl(
            "PROD",
            wells=["P1", "P2"],
            lims={"orat": 8.0e-5},
            allocation="fraction",
            fractions=[0.25, 0.75],
        )
    ]
    ok = facility.update_switched_controls(_state([-5.0e-5, -5.0e-5, 0.0]),
                                           ["qWs", "qOs", "qGs"])

    failures = 0
    failures += _check("generic FacilityModel fraction allocation switches",
                       not ok
                       and np.isclose(facility.wells[0].val, -2.0e-5)
                       and np.isclose(facility.wells[1].val, -6.0e-5))

    facility = _facility()
    facility.group_controls = [
        FacilityGroupControl("PROD", wells=["P1", "P2"], lims={"orat": 6.0e-5}, allocation="equal")
    ]
    ok = facility.update_switched_controls(_state([-1.0e-5, -9.0e-5, 0.0]),
                                           ["qWs", "qOs", "qGs"])
    failures += _check("generic FacilityModel equal allocation switches",
                       not ok
                       and np.isclose(facility.wells[0].val, -3.0e-5)
                       and np.isclose(facility.wells[1].val, -3.0e-5))
    return failures


def validate_nested_guide_group() -> int:
    facility = _facility()
    west = FacilityGroupControl("WEST", wells=["P1", "P2"], lims={}, allocation="equal")
    east = FacilityGroupControl("EAST", wells=["P3"], lims={})
    field = FacilityGroupControl(
        "FIELD",
        wells=["group:WEST", "group:EAST"],
        lims={"orat": 8.0e-5},
        allocation="guide",
        fractions=[1.0, 3.0],
    )
    facility.group_controls = [west, east, field]
    ok = facility.update_switched_controls(_state([-3.0e-5, -3.0e-5, -4.0e-5]),
                                           ["qWs", "qOs", "qGs"])
    report = facility.last_group_switches[0] if facility.last_group_switches else {}

    failures = 0
    failures += _check("generic FacilityModel nested guide allocation switches",
                       not ok
                       and report.get("group") == "FIELD"
                       and report.get("allocation") == "guide")
    failures += _check("nested guide allocation reaches leaf well targets",
                       np.isclose(facility.wells[0].val, -1.0e-5)
                       and np.isclose(facility.wells[1].val, -1.0e-5)
                       and np.isclose(facility.wells[2].val, -6.0e-5))
    failures += _check("nested guide diagnostics expose leaf weights",
                       np.allclose(report.get("allocation_weights"), [0.125, 0.125, 0.75]))
    return failures


def validate_constructor_group_controls() -> int:
    group = FacilityGroupControl("PROD", wells=["P1"], lims={"orat": 1.0e-5})
    facility = FacilityModel([_well("P1")], group_controls=[group])
    failures = 0
    failures += _check("FacilityModel constructor accepts group controls",
                       facility.group_controls == [group])
    return failures


def main() -> int:
    print("\nPhase 288: generic FacilityModel group controls")
    failures = validate_constructor_group_controls()
    failures += validate_rate_group_switch()
    failures += validate_fraction_and_equal_groups()
    failures += validate_nested_guide_group()
    if failures:
        print(f"\nPhase 288 failed ({failures} checks).")
        return 1
    print("\nGeneric FacilityModel now supports aggregate group-control switching.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
