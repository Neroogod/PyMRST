"""Phase 181: MRST WCONPROD phase-composition parity."""

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
    raw = np.arange(1, nph + 1, dtype=float)
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
        compi=raw / raw.sum(),
        refDepth=0.0,
        lims={},
    )


def _apply(mode: str, *, nph: int = 3, name: str = "PPROD") -> Well:
    updated = apply_deck_well_controls(
        [_well(name, nph=nph)],
        {
            "WCONPROD": [[name, "OPEN", mode, 10.0, 4.0, 2.0, 17.0, 5.0, 190.0, 290.0]],
        },
    )
    return updated[0]


def validate_phase_specific_wconprod_compi() -> int:
    orat = _apply("ORAT", name="PORAT")
    wrat = _apply("WRAT", name="PWRAT")
    grat = _apply("GRAT", name="PGRAT")
    lrat = _apply("LRAT", name="PLRAT")
    resv = _apply("RESV", name="PRESV")
    bhp = _apply("BHP", name="PBHP")

    failures = 0
    failures += _check("WCONPROD ORAT uses MRST oil WOG stream",
                       orat.type == "orat" and np.allclose(orat.compi, [0.0, 1.0, 0.0]))
    failures += _check("WCONPROD WRAT uses MRST water WOG stream",
                       wrat.type == "wrat" and np.allclose(wrat.compi, [1.0, 0.0, 0.0]))
    failures += _check("WCONPROD GRAT uses MRST gas WOG stream",
                       grat.type == "grat" and np.allclose(grat.compi, [0.0, 0.0, 1.0]))
    failures += _check("WCONPROD LRAT uses MRST liquid WOG stream",
                       lrat.type == "lrat" and np.allclose(lrat.compi, [0.5, 0.5, 0.0]))
    failures += _check("WCONPROD RESV maps to VRAT with MRST all-phase stream",
                       resv.type == "vrat"
                       and np.isclose(resv.val, -5.0)
                       and np.allclose(resv.compi, np.ones(3) / 3.0))
    failures += _check("WCONPROD BHP uses MRST all-phase placeholder stream",
                       bhp.type == "bhp" and np.allclose(bhp.compi, np.ones(3) / 3.0))
    return failures


def validate_two_component_projection() -> int:
    orat = _apply("ORAT", nph=2, name="TWO_ORAT")
    wrat = _apply("WRAT", nph=2, name="TWO_WRAT")
    grat = _apply("GRAT", nph=2, name="TWO_GRAT")
    lrat = _apply("LRAT", nph=2, name="TWO_LRAT")
    resv = _apply("RESV", nph=2, name="TWO_RESV")
    bhp = _apply("BHP", nph=2, name="TWO_BHP")

    failures = 0
    failures += _check("two-component WCONPROD ORAT projects to oil",
                       np.allclose(orat.compi, [1.0, 0.0]))
    failures += _check("two-component WCONPROD WRAT falls back to oil/gas split",
                       np.allclose(wrat.compi, [0.5, 0.5]))
    failures += _check("two-component WCONPROD GRAT projects to gas",
                       np.allclose(grat.compi, [0.0, 1.0]))
    failures += _check("two-component WCONPROD LRAT projects liquid to oil",
                       np.allclose(lrat.compi, [1.0, 0.0]))
    failures += _check("two-component WCONPROD RESV projects all-phase stream",
                       np.allclose(resv.compi, [0.5, 0.5]))
    failures += _check("two-component WCONPROD BHP projects all-phase stream",
                       np.allclose(bhp.compi, [0.5, 0.5]))
    return failures


def main() -> int:
    print("\nPhase 181: MRST WCONPROD phase-composition parity")
    failures = validate_phase_specific_wconprod_compi()
    failures += validate_two_component_projection()
    if failures:
        print(f"\nPhase 181 failed ({failures} checks).")
        return 1
    print("\nWCONPROD now imports MRST's phase-composition vectors.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
