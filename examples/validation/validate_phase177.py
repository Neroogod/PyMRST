"""Phase 177: MRST WCONHIST historical LRAT target parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.facility import apply_deck_well_controls  # noqa: E402
from pymrst.params.wells_and_bc import Well  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _well(name: str) -> Well:
    return Well(
        cells=np.array([0]),
        type="bhp",
        val=1.0,
        r=np.ones(1),
        dir="z",
        WI=np.ones(1),
        dZ=np.zeros(1),
        name=name,
        compi=np.array([0.0, 1.0, 0.0]),
        refDepth=0.0,
        sign=-1,
        lims={},
    )


def validate_full_wconhist_lrat_uses_historical_oil_water_rates() -> int:
    wells = [_well("PHIST")]
    updated = apply_deck_well_controls(
        wells,
        {
            # MRST process_wconhist ignores the explicit LRAT-like column for
            # historical LRAT controls and uses -(ORAT + WRAT), columns 4/5.
            "WCONHIST": [["PHIST", "OPEN", "LRAT", 10.0, 4.0, 2.0, 999.0, "Default", 0.0, 240.0, 0.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("full WCONHIST LRAT target is negative oil+water history",
                       well.type == "lrat" and np.isclose(well.val, -14.0))
    failures += _check("full WCONHIST LRAT limit is negative oil+water history",
                       np.isclose(well.lims["lrat"], -14.0))
    failures += _check("full WCONHIST oil/water/gas limits remain available",
                       np.isclose(well.lims["orat"], -10.0)
                       and np.isclose(well.lims["wrat"], -4.0)
                       and np.isclose(well.lims["grat"], -2.0))
    return failures


def validate_compact_wconhist_lrat_compatibility() -> int:
    wells = [_well("PCOMPACT")]
    updated = apply_deck_well_controls(
        wells,
        {
            # Older native analogue rows used WCONPROD-like explicit LRAT.
            "WCONHIST": [["PCOMPACT", "OPEN", "LRAT", 10.0, 4.0, 2.0, 17.0, 0.0, 240.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("compact WCONHIST LRAT analogue still uses explicit LRAT",
                       well.type == "lrat" and np.isclose(well.val, -17.0))
    failures += _check("compact WCONHIST LRAT limit remains explicit",
                       np.isclose(well.lims["lrat"], -17.0))
    return failures


def validate_wconprod_lrat_is_unchanged() -> int:
    wells = [_well("PPROD")]
    updated = apply_deck_well_controls(
        wells,
        {
            "WCONPROD": [["PPROD", "OPEN", "LRAT", 10.0, 4.0, 2.0, 17.0, 5.0, 190.0, 290.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("WCONPROD LRAT still uses column 7",
                       well.type == "lrat" and np.isclose(well.val, -17.0))
    failures += _check("WCONPROD LRAT keeps ordinary BHP/THP limits",
                       np.isclose(well.lims["bhp"], 190.0)
                       and np.isclose(well.lims["thp"], 290.0))
    return failures


def main() -> int:
    print("\nPhase 177: MRST WCONHIST historical LRAT target parity")
    failures = validate_full_wconhist_lrat_uses_historical_oil_water_rates()
    failures += validate_compact_wconhist_lrat_compatibility()
    failures += validate_wconprod_lrat_is_unchanged()
    if failures:
        print(f"\nPhase 177 failed ({failures} checks).")
        return 1
    print("\nWCONHIST LRAT rows now use MRST's historical oil+water target.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
