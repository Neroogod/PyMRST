"""Phase 182: MRST WCON VFP-index metadata parity."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.io.mrst_mat import _well_from_dict  # noqa: E402
from pymrst.modules.compositional.facility import apply_deck_well_controls  # noqa: E402
from pymrst.params.wells_and_bc import Well  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _well(name: str, *, sign: int = -1) -> Well:
    return Well(
        cells=np.array([0]),
        type="bhp",
        val=1.0,
        r=np.ones(1),
        dir="z",
        WI=np.ones(1),
        dZ=np.zeros(1),
        name=name,
        sign=sign,
        compi=np.ones(3, dtype=float) / 3.0,
        refDepth=0.0,
        lims={},
    )


def validate_deck_vfp_columns() -> int:
    wells = [_well("P1"), _well("I1", sign=1), _well("I2", sign=1), _well("H1", sign=1)]
    updated = apply_deck_well_controls(
        wells,
        {
            # MRST process_wconprod stores W(end).vfp_index from column 11.
            "WCONPROD": [["P1", "OPEN", "ORAT", 10.0, 4.0, 2.0, 17.0, 5.0, 190.0, 290.0, 7]],
            # MRST process_wconinje stores W(end).vfp_index from column 9.
            "WCONINJE": [["I1", "GAS", "OPEN", "RATE", 3.0, 4.0, 500.0, 600.0, 8]],
            # Legacy WCONINJ stores W(end).vfp_index from column 11.
            "WCONINJ": [["I2", "WATER", "OPEN", "RATE", 3.0, 4.0, "Default", "Default", 500.0, 600.0, 9]],
            # WCONINJH has no MRST vfp_index assignment in processWells.m.
            "WCONINJH": [["H1", "WATER", "OPEN", 3.0, "Default", "RATE", 500.0]],
        },
    )
    by_name = {w.name: w for w in updated}

    failures = 0
    failures += _check("WCONPROD imports MRST column-11 vfp_index",
                       by_name["P1"].vfp_index == 7)
    failures += _check("WCONINJE imports MRST column-9 vfp_index",
                       by_name["I1"].vfp_index == 8)
    failures += _check("legacy WCONINJ imports MRST column-11 vfp_index",
                       by_name["I2"].vfp_index == 9)
    failures += _check("WCONINJH leaves vfp_index unset",
                       by_name["H1"].vfp_index is None)
    return failures


def validate_mapping_aliases_and_missing_values() -> int:
    updated = apply_deck_well_controls(
        [_well("PMAP"), _well("IMAP", sign=1), _well("PNONE")],
        {
            "WCONPROD": [
                {"well": "PMAP", "status": "OPEN", "control": "BHP", "bhp": 120.0, "vfp_index": 3},
                ["PNONE", "OPEN", "BHP", np.inf, np.inf, np.inf, np.inf, np.inf, 100.0],
            ],
            "WCONINJE": [
                {"well": "IMAP", "phase": "GAS", "status": "OPEN", "control": "RATE", "rate": 1.0, "vfp": 4},
            ],
        },
    )
    by_name = {w.name: w for w in updated}

    failures = 0
    failures += _check("mapping WCONPROD vfp_index alias is accepted",
                       by_name["PMAP"].vfp_index == 3)
    failures += _check("mapping WCONINJE vfp alias is accepted",
                       by_name["IMAP"].vfp_index == 4)
    failures += _check("short positional WCONPROD keeps vfp_index unset",
                       by_name["PNONE"].vfp_index is None)
    return failures


def validate_mrst_mat_vfp_index_import() -> int:
    well = _well_from_dict(
        {
            "cells": np.array([1]),
            "type": "bhp",
            "val": 100.0,
            "r": np.array([0.1]),
            "dir": "z",
            "WI": np.array([1.0]),
            "dZ": np.array([0.0]),
            "name": "MAT",
            "compi": np.array([0.0, 1.0, 0.0]),
            "refDepth": 0.0,
            "sign": -1,
            "status": True,
            "vfp_index": 5,
        },
        0,
    )

    failures = 0
    failures += _check("mrst_mat preserves vfp_index on Well",
                       well.vfp_index == 5)
    return failures


def main() -> int:
    print("\nPhase 182: MRST WCON VFP-index metadata parity")
    failures = validate_deck_vfp_columns()
    failures += validate_mapping_aliases_and_missing_values()
    failures += validate_mrst_mat_vfp_index_import()
    if failures:
        print(f"\nPhase 182 failed ({failures} checks).")
        return 1
    print("\nDeck and MATLAB wells now preserve MRST vfp_index metadata.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
