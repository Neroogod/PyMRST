"""Phase 178: MRST WCONINJH historical rate-control parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.facility import apply_deck_well_controls  # noqa: E402
from pymrst.params.wells_and_bc import Well  # noqa: E402
from pymrst.utils.units import BARSA  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _well(name: str, *, nph: int = 3) -> Well:
    compi = np.zeros(nph, dtype=float)
    compi[0] = 1.0
    return Well(
        cells=np.array([0]),
        type="bhp",
        val=1.0,
        r=np.ones(1),
        dir="z",
        WI=np.ones(1),
        dZ=np.zeros(1),
        name=name,
        sign=1,
        compi=compi,
        refDepth=0.0,
        lims={},
    )


def validate_mrst_wconinjh_is_always_rate_controlled() -> int:
    wells = [_well("IHIST")]
    updated = apply_deck_well_controls(
        wells,
        {
            # MRST process_wconinjh: name, phase, status, historical rate.
            "WCONINJH": [["IHIST", "GAS", "OPEN", 17.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("MRST-shaped WCONINJH becomes rate control",
                       well.sign == 1 and well.type == "rate" and np.isclose(well.val, 17.0))
    failures += _check("historical rate is also the RATE limit",
                       well.lims is not None and np.isclose(well.lims["rate"], 17.0))
    failures += _check("MRST-shaped WCONINJH uses the default injector BHP limit",
                       np.isclose(well.lims["bhp"], 6895.0 * BARSA))
    failures += _check("gas WCONINJH stream maps to gas slot",
                       np.allclose(well.compi, [0.0, 0.0, 1.0]))
    return failures


def validate_welltarg_rate_updates_history_rate_column() -> int:
    wells = [_well("ITARG")]
    updated = apply_deck_well_controls(
        wells,
        {
            "WCONINJH": [["ITARG", "WATER", "OPEN", 5.0]],
            "WELTARG": [["ITARG", "RATE", 9.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("WELTARG RATE updates WCONINJH active historical rate",
                       well.type == "rate" and np.isclose(well.val, 9.0))
    failures += _check("WELTARG RATE updates WCONINJH rate limit",
                       np.isclose(well.lims["rate"], 9.0))
    failures += _check("WELTARG RATE keeps MRST default BHP limit",
                       np.isclose(well.lims["bhp"], 6895.0 * BARSA))
    return failures


def validate_legacy_text_mode_row_still_parses() -> int:
    wells = [_well("ILEGACY", nph=2)]
    updated = apply_deck_well_controls(
        wells,
        {
            # Older PyMRST-native analogue rows used a WCONINJE-like control
            # column.  They are not MRST-shaped, but remain accepted so existing
            # native examples do not silently become zero-rate wells.
            "WCONINJH": [["ILEGACY", "WATER", "OPEN", "BHP", 3.0, 4.0, 250.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("legacy textual WCONINJH analogue remains BHP-controlled",
                       well.type == "bhp" and np.isclose(well.val, 250.0))
    failures += _check("legacy textual WCONINJH keeps explicit rate limit",
                       np.isclose(well.lims["rate"], 3.0))
    failures += _check("legacy textual WCONINJH keeps water stream",
                       np.allclose(well.compi, [1.0, 0.0]))
    return failures


def main() -> int:
    print("\nPhase 178: MRST WCONINJH historical rate-control parity")
    failures = validate_mrst_wconinjh_is_always_rate_controlled()
    failures += validate_welltarg_rate_updates_history_rate_column()
    failures += validate_legacy_text_mode_row_still_parses()
    if failures:
        print(f"\nPhase 178 failed ({failures} checks).")
        return 1
    print("\nWCONINJH now follows MRST's historical rate-control semantics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
