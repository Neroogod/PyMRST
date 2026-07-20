"""Phase 179: MRST WCONHIST RESV_HISTORY target parity."""

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
    compi = np.ones(nph, dtype=float) / max(nph, 1)
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
        compi=compi,
        refDepth=0.0,
        lims={},
    )


def validate_full_wconhist_resv_becomes_resv_history() -> int:
    wells = [_well("PHIST")]
    updated = apply_deck_well_controls(
        wells,
        {
            # MRST process_wconhist RESV reads historical ORAT/WRAT/GRAT from
            # columns 4/5/6, negates them, sums the target, and reorders phase
            # fractions from OWG row order to WOG well order.
            "WCONHIST": [["PHIST", "OPEN", "RESV", 10.0, 4.0, 2.0, 999.0, "Default", 0.0, 240.0, 0.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("full WCONHIST RESV becomes RESV_HISTORY control",
                       well.type == "resv_history")
    failures += _check("RESV_HISTORY target is negative sum of historical oil/water/gas",
                       np.isclose(well.val, -16.0)
                       and np.isclose(well.lims["resv_history"], -16.0))
    failures += _check("RESV_HISTORY keeps MRST WOG phase fractions",
                       np.allclose(well.compi, [0.25, 0.625, 0.125]))
    failures += _check("ordinary historical phase limits remain production signed",
                       np.isclose(well.lims["orat"], -10.0)
                       and np.isclose(well.lims["wrat"], -4.0)
                       and np.isclose(well.lims["grat"], -2.0))
    return failures


def validate_zero_history_resv_uses_equal_split() -> int:
    wells = [_well("PZERO")]
    updated = apply_deck_well_controls(
        wells,
        {
            "WCONHIST": [["PZERO", "OPEN", "RESV", 0.0, 0.0, 0.0, 0.0, "Default", 0.0, 200.0, 0.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("zero RESV_HISTORY target remains zero",
                       well.type == "resv_history" and np.isclose(well.val, 0.0))
    failures += _check("zero RESV_HISTORY uses equal WOG split",
                       np.allclose(well.compi, np.ones(3) / 3.0))
    return failures


def validate_two_component_well_keeps_oil_gas_projection() -> int:
    wells = [_well("PTWO", nph=2)]
    updated = apply_deck_well_controls(
        wells,
        {
            "WCONHIST": [["PTWO", "OPEN", "RESV", 10.0, 4.0, 2.0, 0.0, "Default", 0.0, 220.0, 0.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("two-component RESV_HISTORY keeps the MRST target",
                       well.type == "resv_history" and np.isclose(well.val, -16.0))
    failures += _check("two-component RESV_HISTORY projects WOG to oil/gas",
                       np.allclose(well.compi, [10.0 / 12.0, 2.0 / 12.0]))
    return failures


def validate_wconprod_resv_maps_to_vrat() -> int:
    wells = [_well("PPROD")]
    updated = apply_deck_well_controls(
        wells,
        {
            "WCONPROD": [["PPROD", "OPEN", "RESV", 10.0, 4.0, 2.0, 17.0, 5.0, 190.0, 290.0]],
        },
    )
    well = updated[0]

    failures = 0
    failures += _check("ordinary WCONPROD RESV still maps to vrat",
                       well.type == "vrat" and np.isclose(well.val, -5.0))
    failures += _check("ordinary WCONPROD RESV uses MRST all-phase compi",
                       np.allclose(well.compi, np.ones(3) / 3.0))
    return failures


def main() -> int:
    print("\nPhase 179: MRST WCONHIST RESV_HISTORY target parity")
    failures = validate_full_wconhist_resv_becomes_resv_history()
    failures += validate_zero_history_resv_uses_equal_split()
    failures += validate_two_component_well_keeps_oil_gas_projection()
    failures += validate_wconprod_resv_maps_to_vrat()
    if failures:
        print(f"\nPhase 179 failed ({failures} checks).")
        return 1
    print("\nWCONHIST RESV now imports as MRST RESV_HISTORY.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
