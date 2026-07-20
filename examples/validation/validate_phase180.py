"""Phase 180: MRST WCONHIST historical phase-composition parity."""

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


def _well(name: str, *, nph: int = 3) -> Well:
    return Well(
        cells=np.array([0]),
        type="bhp",
        val=1.0,
        r=np.ones(1),
        dir="z",
        WI=np.ones(1),
        dZ=np.zeros(1),
        name=name,
        sign=-1,
        compi=np.ones(nph, dtype=float) / max(nph, 1),
        refDepth=0.0,
        lims={},
    )


def _apply(mode: str, *, nph: int = 3, name: str = "PHIST") -> Well:
    wells = [_well(name, nph=nph)]
    updated = apply_deck_well_controls(
        wells,
        {
            "WCONHIST": [[name, "OPEN", mode, 10.0, 4.0, 2.0, 999.0, "Default", 0.0, 240.0, 0.0]],
        },
    )
    return updated[0]


def validate_phase_specific_history_compi() -> int:
    orat = _apply("ORAT", name="PORAT")
    wrat = _apply("WRAT", name="PWRAT")
    grat = _apply("GRAT", name="PGRAT")
    bhp = _apply("BHP", name="PBHP")

    failures = 0
    failures += _check("WCONHIST ORAT uses MRST oil WOG stream",
                       orat.type == "orat" and np.allclose(orat.compi, [0.0, 1.0, 0.0]))
    failures += _check("WCONHIST WRAT uses MRST water WOG stream",
                       wrat.type == "wrat" and np.allclose(wrat.compi, [1.0, 0.0, 0.0]))
    failures += _check("WCONHIST GRAT uses MRST gas WOG stream",
                       grat.type == "grat" and np.allclose(grat.compi, [0.0, 0.0, 1.0]))
    failures += _check("WCONHIST BHP uses MRST oil WOG stream",
                       bhp.type == "bhp" and np.allclose(bhp.compi, [0.0, 1.0, 0.0]))
    return failures


def validate_lrat_history_compi() -> int:
    well = _apply("LRAT", name="PLRAT")
    zero = apply_deck_well_controls(
        [_well("PZERO")],
        {
            "WCONHIST": [["PZERO", "OPEN", "LRAT", 0.0, 0.0, 2.0, 0.0, "Default", 0.0, 240.0, 0.0]],
        },
    )[0]

    failures = 0
    failures += _check("WCONHIST LRAT uses historical water/oil split",
                       well.type == "lrat"
                       and np.isclose(well.val, -14.0)
                       and np.allclose(well.compi, [4.0 / 14.0, 10.0 / 14.0, 0.0]))
    failures += _check("zero WCONHIST LRAT uses MRST half water/half oil split",
                       zero.type == "lrat"
                       and np.isclose(zero.val, 0.0)
                       and np.allclose(zero.compi, [0.5, 0.5, 0.0]))
    return failures


def validate_two_component_projection() -> int:
    orat = _apply("ORAT", nph=2, name="TWO_ORAT")
    wrat = _apply("WRAT", nph=2, name="TWO_WRAT")
    grat = _apply("GRAT", nph=2, name="TWO_GRAT")
    lrat = _apply("LRAT", nph=2, name="TWO_LRAT")

    failures = 0
    failures += _check("two-component WCONHIST ORAT projects to oil",
                       np.allclose(orat.compi, [1.0, 0.0]))
    failures += _check("two-component WCONHIST WRAT falls back to oil/gas split",
                       np.allclose(wrat.compi, [0.5, 0.5]))
    failures += _check("two-component WCONHIST GRAT projects to gas",
                       np.allclose(grat.compi, [0.0, 1.0]))
    failures += _check("two-component WCONHIST LRAT projects historical oil/gas",
                       np.allclose(lrat.compi, [1.0, 0.0]))
    return failures


def validate_compact_wconhist_compatibility() -> int:
    wells = [_well("PCOMPACT")]
    updated = apply_deck_well_controls(
        wells,
        {
            "WCONHIST": [["PCOMPACT", "OPEN", "ORAT", 10.0, 4.0, 2.0, 17.0, 0.0, 240.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("compact WCONHIST analogue keeps existing composition",
                       np.allclose(well.compi, np.ones(3) / 3.0))
    return failures


def main() -> int:
    print("\nPhase 180: MRST WCONHIST historical phase-composition parity")
    failures = validate_phase_specific_history_compi()
    failures += validate_lrat_history_compi()
    failures += validate_two_component_projection()
    failures += validate_compact_wconhist_compatibility()
    if failures:
        print(f"\nPhase 180 failed ({failures} checks).")
        return 1
    print("\nWCONHIST now imports MRST's historical phase-composition vectors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
