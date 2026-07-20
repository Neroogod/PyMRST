"""Phase 160: GCONPROD/GCONINJE mapping raw-row order parity."""

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


def validate_gconprod_mapping_raw_order() -> int:
    group = facility_group_controls_from_deck_control({
        "GCONPROD": [
            {
                # Deliberately not MRST column order.
                "procedure": "RATE",
                "guide_rate": 42.0,
                "status": "NO",
                "group": "FIELD",
                "control": "ORAT",
                "orat": 1.0e-4,
                "resv": 5.0e-4,
            }
        ]
    })[0]
    raw = group.extra["GCONPROD"][0]["raw"]

    failures = 0
    failures += _check("GCONPROD mapping raw row follows MRST column count",
                       len(raw) == 21)
    failures += _check("GCONPROD mapping raw row starts with group/control/oil target",
                       raw[:3] == ["FIELD", "ORAT", 1.0e-4])
    failures += _check("GCONPROD mapping raw row places status and guide columns correctly",
                       raw[7] == "NO" and np.isclose(raw[8], 42.0))
    failures += _check("GCONPROD mapping raw row places late MRST columns correctly",
                       np.isclose(raw[13], 5.0e-4) and raw[20] == "RATE")
    failures += _check("GCONPROD mapping still keeps native limits",
                       np.isclose(group.lims["orat"], 1.0e-4))
    return failures


def validate_gconinje_mapping_raw_order() -> int:
    group = facility_group_controls_from_deck_control({
        "GCONINJE": [
            {
                # Deliberately not MRST column order.
                "wet_gas": 6.0e-4,
                "guide_rate": 11.0,
                "status": "YES",
                "group": "GINJ",
                "phase": "GAS",
                "control": "RATE",
                "surface_rate": 2.0e-4,
                "reservoir_rate": 3.0e-4,
                "target_pressure": 250.0,
            }
        ]
    })[0]
    raw = group.extra["GCONINJE"][0]["raw"]

    failures = 0
    failures += _check("GCONINJE mapping raw row follows MRST column count",
                       len(raw) == 13)
    failures += _check("GCONINJE mapping raw row starts with group/phase/control",
                       raw[:3] == ["GINJ", "GAS", "RATE"])
    failures += _check("GCONINJE mapping raw row places rate/status/guide columns correctly",
                       np.isclose(raw[3], 2.0e-4)
                       and np.isclose(raw[4], 3.0e-4)
                       and raw[7] == "YES"
                       and np.isclose(raw[8], 11.0))
    failures += _check("GCONINJE mapping raw row places late MRST columns correctly",
                       np.isclose(raw[10], 250.0)
                       and np.isclose(raw[12], 6.0e-4))
    failures += _check("GCONINJE mapping still keeps native limits",
                       np.isclose(group.lims["grat"], 2.0e-4)
                       and np.isclose(group.lims["vrat"], 3.0e-4))
    return failures


def validate_sequence_rows_are_unchanged() -> int:
    row = ["FIELD", "ORAT", 1.0, np.inf, np.inf, np.inf, "NONE", "YES", 5.0]
    group = facility_group_controls_from_deck_control({"GCONPROD": [row]})[0]

    failures = 0
    failures += _check("sequence raw rows remain untouched",
                       group.extra["GCONPROD"][0]["raw"] == row)
    return failures


def main() -> int:
    print("\nPhase 160: GCONPROD/GCONINJE mapping raw-row order parity")
    failures = validate_gconprod_mapping_raw_order()
    failures += validate_gconinje_mapping_raw_order()
    failures += validate_sequence_rows_are_unchanged()
    if failures:
        print(f"\nPhase 160 failed ({failures} checks).")
        return 1
    print("\nMapping-style group control metadata now uses MRST column order.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
