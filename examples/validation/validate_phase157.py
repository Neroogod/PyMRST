"""Phase 157: GCONPROD/GCONINJE metadata preservation."""

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


def validate_gconprod_metadata() -> int:
    control = {
        "GCONPROD": [
            [
                "FIELD", "ORAT", 1.0e-4, 2.0e-4, 3.0e-4, 4.0e-4,
                "CON", "NO", 7.5, "OIL", "NONE", "NONE", "NONE",
                5.0e-4, 0.1, 0.2, 0.3, 0.4, 0.5, 250.0, "RATE",
            ]
        ]
    }
    group = facility_group_controls_from_deck_control(control)[0]
    meta = group.extra.get("GCONPROD", [{}])[0]

    failures = 0
    failures += _check("GCONPROD native limits remain active",
                       np.isclose(group.lims["orat"], 1.0e-4)
                       and np.isclose(group.lims["wrat"], 2.0e-4)
                       and group.status is False)
    failures += _check("GCONPROD metadata preserves control and exceed action",
                       meta.get("control") == "orat"
                       and meta.get("exceed_action") == "CON"
                       and meta.get("procedure") == "RATE")
    failures += _check("GCONPROD metadata preserves guide fields",
                       np.isclose(meta.get("guide_rate"), 7.5)
                       and meta.get("guide_phase") == "OIL")
    failures += _check("GCONPROD metadata preserves late numeric columns",
                       np.isclose(meta.get("resv"), 5.0e-4)
                       and np.isclose(meta.get("trgp"), 250.0))
    failures += _check("GCONPROD metadata preserves raw row",
                       meta.get("raw", [None])[0] == "FIELD"
                       and meta.get("raw", [None])[20] == "RATE")
    return failures


def validate_gconinje_metadata() -> int:
    control = {
        "GCONINJE": [
            [
                "GINJ", "GAS", "RATE", 1.5e-4, 2.5e-4, 3.5e-4,
                4.5e-4, "YES", 9.5, "GAS", 210.0, 0.75, 6.5e-4,
            ]
        ]
    }
    group = facility_group_controls_from_deck_control(control)[0]
    meta = group.extra.get("GCONINJE", [{}])[0]

    failures = 0
    failures += _check("GCONINJE native gas limit remains active",
                       np.isclose(group.lims["grat"], 1.5e-4)
                       and np.isclose(group.lims["vrat"], 2.5e-4)
                       and group.status is True)
    failures += _check("GCONINJE metadata preserves control and phase",
                       meta.get("phase") == "GAS"
                       and meta.get("control") == "rate")
    failures += _check("GCONINJE metadata preserves guide and target fields",
                       np.isclose(meta.get("guide_rate"), 9.5)
                       and meta.get("guide_phase") == "GAS"
                       and np.isclose(meta.get("trgp"), 210.0))
    failures += _check("GCONINJE metadata preserves reinjection/voidage fields",
                       np.isclose(meta.get("reinject_target"), 3.5e-4)
                       and np.isclose(meta.get("voidage_target"), 4.5e-4)
                       and np.isclose(meta.get("vrep"), 0.75))
    failures += _check("GCONINJE metadata preserves raw row",
                       meta.get("raw", [None])[0] == "GINJ"
                       and np.isclose(meta.get("raw", [0])[12], 6.5e-4))
    return failures


def validate_schedule_metadata_conversion() -> int:
    schedule = {
        "control": [
            {"GCONPROD": [["FIELD", "ORAT", 1.0, np.inf, np.inf, np.inf, "NONE", "YES", 2.0]]},
            {"GCONINJE": [["GINJ", "WATER", "RATE", 3.0, np.inf, np.inf, np.inf, "NO", 4.0]]},
        ]
    }
    converted = facility_group_controls_from_deck_schedule(schedule)

    failures = 0
    failures += _check("schedule conversion preserves GCONPROD metadata",
                       converted[0][0].extra["GCONPROD"][0]["guide_rate"] == 2.0)
    failures += _check("schedule conversion preserves GCONINJE metadata",
                       converted[1][0].extra["GCONINJE"][0]["status"] is False
                       and converted[1][0].extra["GCONINJE"][0]["guide_rate"] == 4.0)
    return failures


def main() -> int:
    print("\nPhase 157: GCONPROD/GCONINJE metadata preservation")
    failures = validate_gconprod_metadata()
    failures += validate_gconinje_metadata()
    failures += validate_schedule_metadata_conversion()
    if failures:
        print(f"\nPhase 157 failed ({failures} checks).")
        return 1
    print("\nImported GCONPROD/GCONINJE rows now preserve MRST control metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
