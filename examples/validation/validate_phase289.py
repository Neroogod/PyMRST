"""Phase 289: selectFacilityFromDeck imports group controls."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.ad_core.models.facilities import (  # noqa: E402
    FacilityModel,
    group_controls_from_deck,
    select_facility_from_deck_ad,
)
from pymrst.modules.ad_core.statefunctions.state_function import State  # noqa: E402
from pymrst.params.wells_and_bc import Well  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _well(name: str) -> Well:
    return Well(
        cells=np.array([0]),
        type="bhp",
        val=100.0,
        r=np.ones(1),
        dir="z",
        WI=np.ones(1),
        dZ=np.zeros(1),
        name=name,
        sign=-1,
        compi=np.array([0.0, 1.0, 0.0]),
        refDepth=0.0,
        lims={},
    )


def _deck() -> dict:
    return {
        "SCHEDULE": {
            "control": [
                {
                    "WELSPECS": [["P1", "PROD"], ["P2", "PROD"], ["P3", "PROD"]],
                    "GCONPROD": [["PROD", "ORAT", 7.5e-5, np.inf, np.inf, np.inf, "NONE", "YES"]],
                    "WGRUPCON": [
                        ["P1", "YES", 1.0, "PROD", 2.0],
                        ["P2", "YES", 3.0, "PROD", 0.5],
                        ["P3", "YES", 4.0, "PROD", 1.0],
                    ],
                },
                {
                    "WELSPECS": [["P1", "PROD"], ["P2", "PROD"]],
                    "GCONPROD": [["PROD", "ORAT", 6.0e-5, np.inf, np.inf, np.inf, "NONE", "YES"]],
                    "WGRUPCON": [
                        ["P1", "YES", 1.0, "PROD"],
                        ["P2", "YES", 1.0, "PROD"],
                    ],
                },
            ]
        }
    }


class ModelShell:
    def __init__(self, wells, facility=None):
        self.wells = list(wells)
        self.FacilityModel = facility


def _state() -> State:
    return State(
        {
            "bhp": np.array([90.0, 91.0, 92.0], dtype=float),
            "qWs": np.zeros(3),
            "qOs": np.array([-5.0e-5, -5.0e-5, -5.0e-5], dtype=float),
            "qGs": np.zeros(3),
        }
    )


def validate_group_schedule_collector() -> int:
    schedule = group_controls_from_deck(_deck())
    failures = 0
    failures += _check("group_controls_from_deck returns one group list per control",
                       len(schedule) == 2
                       and len(schedule[0]) == 1
                       and len(schedule[1]) == 1)
    failures += _check("deck WGRUPCON guides survive facility group import",
                       schedule[0][0].allocation == "fraction"
                       and np.allclose(schedule[0][0].fractions, [2.0, 1.5, 4.0]))
    return failures


def validate_new_facility_receives_deck_groups() -> int:
    wells = [_well("P1"), _well("P2"), _well("P3")]
    model = ModelShell(wells)
    facility = select_facility_from_deck_ad(_deck(), model)
    ok = facility.update_switched_controls(_state(), ["qWs", "qOs", "qGs"])

    failures = 0
    failures += _check("selectFacilityFromDeck attaches group schedule",
                       model.FacilityModel is facility
                       and hasattr(facility, "group_control_schedule")
                       and len(facility.group_control_schedule) == 2)
    failures += _check("selectFacilityFromDeck activates first control groups",
                       len(facility.group_controls) == 1
                       and facility.group_controls[0].name == "PROD")
    failures += _check("imported group control switches generic facility",
                       not ok
                       and np.allclose(
                           [facility.wells[0].val, facility.wells[1].val, facility.wells[2].val],
                           [-2.0e-5, -1.5e-5, -4.0e-5],
                       ))
    return failures


def validate_existing_facility_receives_deck_groups() -> int:
    wells = [_well("P1"), _well("P2"), _well("P3")]
    facility = FacilityModel(wells)
    model = ModelShell([], facility=facility)
    returned = select_facility_from_deck_ad(_deck(), model)

    failures = 0
    failures += _check("selectFacilityFromDeck reuses existing facility for group import",
                       returned is facility)
    failures += _check("existing facility receives deck group controls",
                       len(facility.group_controls) == 1
                       and facility.group_controls[0].name == "PROD"
                       and len(facility.group_control_schedule) == 2)
    return failures


def main() -> int:
    print("\nPhase 289: selectFacilityFromDeck group controls")
    failures = validate_group_schedule_collector()
    failures += validate_new_facility_receives_deck_groups()
    failures += validate_existing_facility_receives_deck_groups()
    if failures:
        print(f"\nPhase 289 failed ({failures} checks).")
        return 1
    print("\nselectFacilityFromDeck now carries deck group controls into native facilities.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
