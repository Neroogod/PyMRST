"""Phase 176: MRST WCONHIST BHP column layout parity."""

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


def validate_mrst_wconhist_bhp_column() -> int:
    wells = [_well("PHIST")]
    updated = apply_deck_well_controls(
        wells,
        {
            # MRST process_wconhist reads BHP-controlled history producers from
            # column 10, not WCONPROD's column 9 BHP slot.
            "WCONHIST": [["PHIST", "OPEN", "BHP", 1.0, 2.0, 3.0, 0.0, "Default", 0.0, 240.0, 0.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("MRST-shaped WCONHIST BHP reads column 10",
                       well.type == "bhp" and np.isclose(well.val, 240.0))
    failures += _check("MRST-shaped WCONHIST does not mirror BHP into THP",
                       np.isclose(well.lims["bhp"], 240.0)
                       and "thp" not in well.lims)
    return failures


def validate_compact_wconhist_bhp_compatibility() -> int:
    wells = [_well("PCOMPACT")]
    updated = apply_deck_well_controls(
        wells,
        {
            # Compact native analogue rows used WCONPROD-like BHP position.
            "WCONHIST": [["PCOMPACT", "OPEN", "BHP", 1.0, 2.0, 3.0, 0.0, 0.0, 210.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("compact WCONHIST analogue remains accepted",
                       well.type == "bhp" and np.isclose(well.val, 210.0))
    failures += _check("compact WCONHIST keeps BHP-only limit",
                       np.isclose(well.lims["bhp"], 210.0)
                       and "thp" not in well.lims)
    return failures


def validate_weltarg_updates_full_wconhist_bhp() -> int:
    wells = [_well("PTARG")]
    updated = apply_deck_well_controls(
        wells,
        {
            "WCONHIST": [["PTARG", "OPEN", "BHP", 1.0, 2.0, 3.0, 0.0, "Default", 0.0, 240.0, 0.0]],
            "WELTARG": [["PTARG", "BHP", 180.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("WELTARG updates MRST-shaped WCONHIST BHP slot",
                       well.type == "bhp" and np.isclose(well.val, 180.0))
    failures += _check("WELTARG leaves compact-independent WCONHIST limits clean",
                       np.isclose(well.lims["bhp"], 180.0)
                       and "thp" not in well.lims)
    return failures


def validate_wconprod_bhp_column_is_unchanged() -> int:
    wells = [_well("PPROD")]
    updated = apply_deck_well_controls(
        wells,
        {
            "WCONPROD": [["PPROD", "OPEN", "BHP", 1.0, 2.0, 3.0, 4.0, 5.0, 190.0, 290.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("WCONPROD still reads BHP from column 9",
                       well.type == "bhp" and np.isclose(well.val, 190.0))
    failures += _check("WCONPROD still preserves THP from column 10",
                       np.isclose(well.lims["bhp"], 190.0)
                       and np.isclose(well.lims["thp"], 290.0))
    return failures


def main() -> int:
    print("\nPhase 176: MRST WCONHIST BHP column layout parity")
    failures = validate_mrst_wconhist_bhp_column()
    failures += validate_compact_wconhist_bhp_compatibility()
    failures += validate_weltarg_updates_full_wconhist_bhp()
    failures += validate_wconprod_bhp_column_is_unchanged()
    if failures:
        print(f"\nPhase 176 failed ({failures} checks).")
        return 1
    print("\nWCONHIST BHP rows now use MRST history-control columns without changing WCONPROD.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
