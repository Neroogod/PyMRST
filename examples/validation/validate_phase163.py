"""Phase 163: MRST-style well-control keyword table replacement."""

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


def validate_same_keyword_last_row_wins() -> int:
    well = apply_deck_well_controls(
        [_well("P1")],
        {
            "WCONPROD": [
                ["P1", "OPEN", "ORAT", 10.0, np.inf, np.inf, np.inf, np.inf, 100.0],
                ["P1", "SHUT", "GRAT", np.inf, np.inf, 30.0, np.inf, np.inf, 90.0],
            ]
        },
    )[0]

    failures = 0
    failures += _check("same-keyword WCONPROD keeps last row",
                       well.sign == -1
                       and well.status is False
                       and well.type == "grat"
                       and np.isclose(well.val, -30.0))
    failures += _check("same-keyword WCONPROD drops earlier limits",
                       "orat" not in well.lims and np.isclose(well.lims["bhp"], 90.0))
    return failures


def validate_later_injector_keyword_excludes_producer() -> int:
    well = apply_deck_well_controls(
        [_well("MIXED")],
        {
            "WCONPROD": [["MIXED", "OPEN", "ORAT", 10.0, np.inf, np.inf, np.inf, np.inf, 100.0]],
            "WCONINJE": [["MIXED", "GAS", "OPEN", "RATE", 20.0, np.inf, 200.0]],
            "WELTARG": [["MIXED", "BHP", 180.0]],
        },
    )[0]

    failures = 0
    failures += _check("later injector keyword removes producer row",
                       well.sign == 1 and well.type == "rate" and np.isclose(well.val, 20.0))
    failures += _check("WELTARG applies to surviving injector table",
                       np.isclose(well.lims["bhp"], 180.0))
    failures += _check("later injector stream controls composition",
                       np.allclose(well.compi, [0.0, 1.0]))
    return failures


def validate_later_producer_keyword_excludes_injector() -> int:
    well = apply_deck_well_controls(
        [_well("MIXED")],
        {
            "WCONINJE": [["MIXED", "WATER", "OPEN", "RATE", 20.0, np.inf, 200.0]],
            "WCONPROD": [["MIXED", "OPEN", "BHP", np.inf, np.inf, np.inf, np.inf, np.inf, 100.0]],
            "WELTARG": [["MIXED", "BHP", 90.0]],
        },
    )[0]

    failures = 0
    failures += _check("later producer keyword removes injector row",
                       well.sign == -1 and well.type == "bhp" and np.isclose(well.val, 90.0))
    failures += _check("surviving producer keeps production status",
                       well.status is True and np.isclose(well.lims["bhp"], 90.0))
    return failures


def validate_case_insensitive_replacement() -> int:
    well = apply_deck_well_controls(
        [_well("P1")],
        {
            "WCONHIST": [
                ["p1", "OPEN", "ORAT", 10.0, np.inf, np.inf, np.inf, np.inf, 100.0],
                ["P1", "OPEN", "WRAT", np.inf, 15.0, np.inf, np.inf, np.inf, 95.0],
            ]
        },
    )[0]

    failures = 0
    failures += _check("same-keyword duplicate matching is case-insensitive",
                       well.sign == -1 and well.type == "wrat" and np.isclose(well.val, -15.0))
    failures += _check("replacement row's well name still maps to existing geometry",
                       well.name == "P1")
    return failures


def main() -> int:
    print("\nPhase 163: MRST-style well-control keyword table replacement")
    failures = validate_same_keyword_last_row_wins()
    failures += validate_later_injector_keyword_excludes_producer()
    failures += validate_later_producer_keyword_excludes_injector()
    failures += validate_case_insensitive_replacement()
    if failures:
        print(f"\nPhase 163 failed ({failures} checks).")
        return 1
    print("\nDeck well-control tables now mirror MRST replacement/exclusion rules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
