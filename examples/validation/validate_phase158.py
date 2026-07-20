"""Phase 158: GCONPROD positional guide rate is not a total-rate limit."""

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


def validate_positional_gconprod_guide_is_metadata_only() -> int:
    groups = facility_group_controls_from_deck_control(
        {
            # MRST readGConProd column 9 is guide rate, not a production target.
            "GCONPROD": [["FIELD", "ORAT", 1.0e-4, np.inf, np.inf, np.inf, "NONE", "YES", 42.0, "OIL"]],
        }
    )
    field = groups[0]
    meta = field.extra["GCONPROD"][0]

    failures = 0
    failures += _check("positional GCONPROD oil target remains active",
                       np.isclose(field.lims["orat"], 1.0e-4))
    failures += _check("positional GCONPROD guide rate is not native rate limit",
                       "rate" not in field.lims)
    failures += _check("positional GCONPROD guide rate is preserved as metadata",
                       np.isclose(meta["guide_rate"], 42.0)
                       and meta["guide_phase"] == "OIL")
    return failures


def validate_mapping_total_rate_alias_still_supported() -> int:
    groups = facility_group_controls_from_deck_control(
        {
            "GCONPROD": [
                {
                    "group": "FIELD",
                    "control": "RATE",
                    "target": 2.5e-4,
                    "guide_rate": 12.0,
                }
            ],
        }
    )
    field = groups[0]

    failures = 0
    failures += _check("mapping GCONPROD total target still becomes rate limit",
                       np.isclose(field.lims["rate"], 2.5e-4))
    failures += _check("mapping GCONPROD guide rate remains metadata",
                       np.isclose(field.extra["GCONPROD"][0]["guide_rate"], 12.0))
    return failures


def main() -> int:
    print("\nPhase 158: GCONPROD guide-rate column parity")
    failures = validate_positional_gconprod_guide_is_metadata_only()
    failures += validate_mapping_total_rate_alias_still_supported()
    if failures:
        print(f"\nPhase 158 failed ({failures} checks).")
        return 1
    print("\nGCONPROD positional guide rates no longer create false total-rate limits.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
