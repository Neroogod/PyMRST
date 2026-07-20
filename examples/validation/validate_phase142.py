"""Phase 142: deck-imported compositional group economics metadata."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.facility import (  # noqa: E402
    facility_group_controls_from_deck_control,
    facility_group_controls_from_deck_schedule,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def validate_gecon_and_grupnet_metadata() -> int:
    control = {
        "WELSPECS": [["P1", "FIELD"]],
        "GCONPROD": [["FIELD", "ORAT", 9.0e-5, np.inf, np.inf, np.inf, "NONE", "YES"]],
        # GECON template in MRST readWellKW:
        # name, min oil, min gas, max water cut, max GOR, max WGR, workover,
        # end-run flag, follow-on group.
        "GECON": [["FIELD", 1.0e-5, 2.0e-5, 0.85, 300.0, 0.2, "CON", "YES", 7]],
        # GRUPNET template in MRST readWellKW:
        # name, guide rate, min VFP table, max VFP table, network flag,
        # VFP flag, terminal node.
        "GRUPNET": [["FIELD", 4.0e-5, 11, 22, "YES", "NO", "NODE_A"]],
    }
    groups = facility_group_controls_from_deck_control(control)
    field = {g.name: g for g in groups}["FIELD"]
    gecon = field.extra.get("GECON", {})
    grupnet = field.extra.get("GRUPNET", {})

    failures = 0
    failures += _check("normal GCONPROD limit remains active with metadata",
                       np.isclose(field.lims["orat"], 9.0e-5))
    failures += _check("GECON economics row is preserved",
                       np.isclose(gecon.get("min_oil_rate"), 1.0e-5)
                       and np.isclose(gecon.get("max_water_cut"), 0.85)
                       and gecon.get("workover") == "CON")
    failures += _check("GECON end-run flag is decoded",
                       gecon.get("end_run") is True)
    failures += _check("GRUPNET network row is preserved",
                       np.isclose(grupnet.get("guide_rate"), 4.0e-5)
                       and grupnet.get("use_network") is True
                       and grupnet.get("use_vfp") is False
                       and grupnet.get("terminal") == "NODE_A")
    failures += _check("raw metadata rows remain available for exact ports",
                       gecon.get("raw", [None])[0] == "FIELD"
                       and grupnet.get("raw", [None])[0] == "FIELD")
    return failures


def validate_metadata_only_group_is_not_dropped() -> int:
    control = {
        "GRUPTREE": [["SATELLITE", "FIELD"]],
        "GECON": [["SATELLITE", 1.0, 2.0, 3.0, 4.0, 5.0, "NONE", "NO", 0]],
    }
    groups = facility_group_controls_from_deck_control(control)
    names = [g.name for g in groups]
    satellite = {g.name: g for g in groups}["SATELLITE"]

    failures = 0
    failures += _check("metadata-only child group is preserved",
                       "SATELLITE" in names)
    failures += _check("metadata-only group keeps empty membership",
                       satellite.wells == [])
    failures += _check("metadata-only group carries GECON metadata",
                       "GECON" in satellite.extra)
    return failures


def validate_schedule_metadata_conversion() -> int:
    schedule = {
        "control": [
            {"WELSPECS": [["P1", "FIELD"]], "GRUPNET": [["FIELD", 1.0, 0, 0, "YES", "YES", "T1"]]},
            {"WELSPECS": [["P1", "FIELD"]], "GECON": [["FIELD", 2.0, 0, 0, 0, 0, "NONE", "NO", 0]]},
        ]
    }
    converted = facility_group_controls_from_deck_schedule(schedule)

    failures = 0
    failures += _check("schedule conversion preserves control count",
                       len(converted) == 2)
    failures += _check("first control carries GRUPNET metadata",
                       "GRUPNET" in converted[0][0].extra)
    failures += _check("second control carries GECON metadata",
                       "GECON" in converted[1][0].extra)
    return failures


def main() -> int:
    print("\nPhase 142: deck-imported compositional group economics metadata")
    failures = validate_gecon_and_grupnet_metadata()
    failures += validate_metadata_only_group_is_not_dropped()
    failures += validate_schedule_metadata_conversion()
    if failures:
        print(f"\nPhase 142 failed ({failures} checks).")
        return 1
    print("\nGECON and GRUPNET rows are preserved on imported group controls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
