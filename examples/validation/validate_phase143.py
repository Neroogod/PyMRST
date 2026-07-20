"""Phase 143: MRST well-control keyword aliases for compositional decks."""

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


def _well(name: str, *, compi=None) -> Well:
    return Well(
        cells=np.array([0]),
        type="bhp",
        val=1.0,
        r=np.ones(1),
        dir="z",
        WI=np.ones(1),
        dZ=np.zeros(1),
        name=name,
        compi=np.asarray([0.5, 0.5] if compi is None else compi, dtype=float),
        refDepth=0.0,
        lims={},
    )


def validate_wconhist_alias() -> int:
    wells = [_well("PHIST")]
    updated = apply_deck_well_controls(
        wells,
        {
            # WCONHIST follows the production-control row shape in MRST.
            "WCONHIST": [["PHIST", "OPEN", "GRAT", 10.0, 20.0, 30.0, 40.0, 50.0, 200.0]],
            "WELTARG": [["PHIST", "BHP", 175.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("WCONHIST is treated as producer control",
                       well.sign == -1 and well.type == "grat")
    failures += _check("WCONHIST active rate target is production signed",
                       np.isclose(well.val, -30.0))
    failures += _check("WELTARG applies to WCONHIST records",
                       well.lims is not None and np.isclose(well.lims["bhp"], 175.0))
    return failures


def validate_wconinjh_alias() -> int:
    wells = [_well("IHIST")]
    updated = apply_deck_well_controls(
        wells,
        {
            # MRST WCONINJH has no control-mode column: column 4 is history rate.
            "WCONINJH": [["IHIST", "WATER", "OPEN", 11.0]],
            "WELLTARG": [["IHIST", "RATE", 12.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("WCONINJH is treated as injector control",
                       well.sign == 1 and well.type == "rate")
    failures += _check("WCONINJH historical rate is active value",
                       np.isclose(well.val, 12.0))
    failures += _check("WCONINJH uses MRST default BHP limit",
                       well.lims is not None
                       and np.isclose(well.lims["rate"], 12.0)
                       and np.isclose(well.lims["bhp"], 6895.0 * BARSA))
    failures += _check("WCONINJH water stream maps to first phase slot",
                       np.allclose(well.compi, [1.0, 0.0]))
    return failures


def validate_wconinj_alias() -> int:
    wells = [_well("IOLD", compi=[1.0, 0.0, 0.0])]
    updated = apply_deck_well_controls(
        wells,
        {
            "WCONINJ": [["IOLD", "OIL", "OPEN", "RATE", 15.0, 25.0, 350.0, 450.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("WCONINJ legacy keyword is accepted",
                       well.sign == 1 and well.type == "rate" and np.isclose(well.val, 15.0))
    failures += _check("WCONINJ oil stream maps to oil slot in three-phase compi",
                       np.allclose(well.compi, [0.0, 1.0, 0.0]))
    failures += _check("WCONINJ preserves BHP/THP limits",
                       well.lims is not None
                       and np.isclose(well.lims["bhp"], 350.0)
                       and np.isclose(well.lims["thp"], 450.0))
    return failures


def main() -> int:
    print("\nPhase 143: MRST well-control keyword aliases")
    failures = validate_wconhist_alias()
    failures += validate_wconinjh_alias()
    failures += validate_wconinj_alias()
    if failures:
        print(f"\nPhase 143 failed ({failures} checks).")
        return 1
    print("\nWCONHIST, WCONINJH, and WCONINJ update native wells.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
