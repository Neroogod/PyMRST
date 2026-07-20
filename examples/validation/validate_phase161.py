"""Phase 161: GECON/GRUPNET/WGRUPCON mapping raw-row order parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.facility import (  # noqa: E402
    facility_group_controls_from_deck_control,
)


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def validate_gecon_mapping_raw_order() -> int:
    group = facility_group_controls_from_deck_control({
        "GECON": [
            {
                # Deliberately not MRST column order.
                "follow_on": 7,
                "end_run": "YES",
                "group": "FIELD",
                "min_oil_rate": 1.0,
                "max_gor": 2.0,
                "workover": "CON",
            }
        ]
    })[0]
    raw = group.extra["GECON"]["raw"]

    failures = 0
    failures += _check("GECON mapping raw row follows MRST column count",
                       len(raw) == 9)
    failures += _check("GECON mapping raw row starts with group/min oil",
                       raw[:2] == ["FIELD", 1.0])
    failures += _check("GECON mapping raw row places GOR/workover/end-run correctly",
                       np.isclose(raw[4], 2.0)
                       and raw[6] == "CON"
                       and raw[7] == "YES"
                       and raw[8] == 7)
    return failures


def validate_grupnet_mapping_raw_order() -> int:
    group = facility_group_controls_from_deck_control({
        "GRUPNET": [
            {
                # Deliberately not MRST column order.
                "terminal": "NODE1",
                "use_vfp": "YES",
                "group": "FIELD",
                "guide_rate": 11.0,
                "max_vfp_table": 4.0,
                "use_network": "YES",
            }
        ]
    })[0]
    raw = group.extra["GRUPNET"]["raw"]

    failures = 0
    failures += _check("GRUPNET mapping raw row follows MRST column count",
                       len(raw) == 7)
    failures += _check("GRUPNET mapping raw row places guide/VFP/network fields",
                       raw == ["FIELD", 11.0, 0.0, 4.0, "YES", "YES", "NODE1"])
    failures += _check("GRUPNET parsed metadata remains available",
                       group.extra["GRUPNET"]["use_network"] is True
                       and np.isclose(group.extra["GRUPNET"]["guide_rate"], 11.0))
    return failures


def validate_wgrupcon_mapping_raw_order() -> int:
    group = facility_group_controls_from_deck_control({
        "WGRUPCON": [
            {
                # Deliberately not MRST column order.
                "guide_multiplier": 3.0,
                "group": "FIELD",
                "well": "P1",
                "status": "YES",
                "guide_rate": 2.0,
            }
        ],
        "GCONPROD": [["FIELD", "ORAT", 1.0e-4]],
    })[0]
    raw = group.extra["WGRUPCON"][0]["raw"]

    failures = 0
    failures += _check("WGRUPCON mapping raw row follows MRST column count",
                       len(raw) == 5)
    failures += _check("WGRUPCON mapping raw row places guide multiplier",
                       raw == ["P1", "YES", 2.0, "FIELD", 3.0])
    failures += _check("WGRUPCON effective guide uses multiplier",
                       np.isclose(group.extra["WGRUPCON"][0]["effective_guide_rate"], 6.0))
    return failures


def validate_sequence_rows_are_unchanged() -> int:
    row = ["P1", "YES", 2.0, "FIELD", 3.0]
    group = facility_group_controls_from_deck_control({
        "WGRUPCON": [row],
        "GCONPROD": [["FIELD", "ORAT", 1.0e-4]],
    })[0]

    failures = 0
    failures += _check("sequence raw rows remain untouched",
                       group.extra["WGRUPCON"][0]["raw"] == row)
    return failures


def main() -> int:
    print("\nPhase 161: GECON/GRUPNET/WGRUPCON mapping raw-row order parity")
    failures = validate_gecon_mapping_raw_order()
    failures += validate_grupnet_mapping_raw_order()
    failures += validate_wgrupcon_mapping_raw_order()
    failures += validate_sequence_rows_are_unchanged()
    if failures:
        print(f"\nPhase 161 failed ({failures} checks).")
        return 1
    print("\nMapping-style economics/network/guide metadata now uses MRST column order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
