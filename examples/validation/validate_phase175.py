"""Phase 175: MRST legacy WCONINJ column layout parity."""

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
        compi=np.array([0.5, 0.5, 0.0]),
        refDepth=0.0,
        sign=1,
        lims={},
    )


def validate_mrst_wconinj_columns() -> int:
    wells = [_well("IOLD")]
    updated = apply_deck_well_controls(
        wells,
        {
            # MRST WCONINJ columns: name, phase, status, control, rate, resv,
            # voidage, VFP-table, BHP, THP, VFP-index, ...
            "WCONINJ": [["IOLD", "OIL", "OPEN", "RATE", 15.0, 25.0, 0.0, "NONE", 350.0, 450.0, 7.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("MRST-shaped WCONINJ keeps active rate from column 5",
                       well.type == "rate" and np.isclose(well.val, 15.0))
    failures += _check("MRST-shaped WCONINJ reads BHP/THP from columns 9/10",
                       np.isclose(well.lims["bhp"], 350.0)
                       and np.isclose(well.lims["thp"], 450.0))
    failures += _check("MRST-shaped WCONINJ keeps phase stream mapping",
                       np.allclose(well.compi, [0.0, 1.0, 0.0]))
    return failures


def validate_compact_wconinj_compatibility() -> int:
    wells = [_well("ICOMPACT")]
    updated = apply_deck_well_controls(
        wells,
        {"WCONINJ": [["ICOMPACT", "GAS", "OPEN", "RATE", 11.0, 22.0, 333.0, 444.0]]},
    )
    well = updated[0]

    failures = 0
    failures += _check("compact WCONINJ analogue remains accepted",
                       well.type == "rate" and np.isclose(well.val, 11.0))
    failures += _check("compact WCONINJ analogue keeps old BHP/THP columns",
                       np.isclose(well.lims["bhp"], 333.0)
                       and np.isclose(well.lims["thp"], 444.0))
    failures += _check("compact WCONINJ gas stream maps to last component",
                       np.allclose(well.compi, [0.0, 0.0, 1.0]))
    return failures


def validate_wconinje_full_length_is_unchanged() -> int:
    wells = [_well("INEW")]
    updated = apply_deck_well_controls(
        wells,
        {
            # Full WCONINJE rows keep BHP/THP at columns 7/8, even when later
            # columns are present for VFP/table metadata.
            "WCONINJE": [["INEW", "WATER", "OPEN", "RATE", 12.0, 24.0, 360.0, 460.0, 9.0, 0.0, 0.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("full WCONINJE keeps active rate from column 5",
                       well.type == "rate" and np.isclose(well.val, 12.0))
    failures += _check("full WCONINJE still reads BHP/THP from columns 7/8",
                       np.isclose(well.lims["bhp"], 360.0)
                       and np.isclose(well.lims["thp"], 460.0))
    failures += _check("full WCONINJE water stream maps to first component",
                       np.allclose(well.compi, [1.0, 0.0, 0.0]))
    return failures


def main() -> int:
    print("\nPhase 175: MRST legacy WCONINJ column layout parity")
    failures = validate_mrst_wconinj_columns()
    failures += validate_compact_wconinj_compatibility()
    failures += validate_wconinje_full_length_is_unchanged()
    if failures:
        print(f"\nPhase 175 failed ({failures} checks).")
        return 1
    print("\nLegacy WCONINJ rows now use MRST column positions without breaking WCONINJE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
