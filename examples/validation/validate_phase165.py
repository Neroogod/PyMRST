"""Phase 165: MRST-style completion status (`cstatus`) support."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.modules.compositional.facility import apply_deck_well_controls  # noqa: E402
from pymrst.modules.compositional.model import OverallCompositionModel  # noqa: E402
from pymrst.params.wells_and_bc import Well  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _well(name: str) -> Well:
    return Well(
        cells=np.array([0, 1, 2]),
        type="bhp",
        val=1.0,
        r=np.ones(3),
        dir="z",
        WI=np.array([10.0, 20.0, 30.0]),
        dZ=np.zeros(3),
        name=name,
        compi=np.asarray([0.5, 0.5], dtype=float),
        refDepth=0.0,
        lims={},
    )


def validate_welopen_completion_range_sets_cstatus() -> int:
    well = apply_deck_well_controls(
        [_well("P1")],
        {
            "WCONPROD": [["P1", "OPEN", "BHP", np.inf, np.inf, np.inf, np.inf, np.inf, 100.0]],
            "WELOPEN": [["P1", "SHUT", 0, 0, 0, 2, 3]],
        },
    )[0]

    failures = 0
    failures += _check("wildcard-IJK WELOPEN completion range sets cstatus",
                       np.array_equal(well.cstatus, np.array([True, False, False])))
    failures += _check("completion WELOPEN does not overwrite whole-well status",
                       well.status is True)
    return failures


def validate_welopen_completion_reopen_preserves_wi() -> int:
    base = _well("P2")
    base.cstatus = np.array([True, False, False])
    well = apply_deck_well_controls(
        [base],
        {
            "WCONPROD": [["P2", "OPEN", "BHP", np.inf, np.inf, np.inf, np.inf, np.inf, 100.0]],
            "WELOPEN": [["P2", "OPEN", 0, 0, 0, 3, 3]],
        },
    )[0]

    failures = 0
    failures += _check("completion WELOPEN can reopen selected perforations",
                       np.array_equal(well.cstatus, np.array([True, False, True])))
    failures += _check("original WI values are preserved when cstatus changes",
                       np.allclose(well.WI, [10.0, 20.0, 30.0]))
    return failures


def validate_positive_ijk_is_conservative_noop_without_grid_lookup() -> int:
    well = apply_deck_well_controls(
        [_well("P3")],
        {
            "WCONPROD": [["P3", "OPEN", "BHP", np.inf, np.inf, np.inf, np.inf, np.inf, 100.0]],
            "WELOPEN": [["P3", "SHUT", 1, 1, 1, 1, 3]],
        },
    )[0]

    return _check(
        "positive-IJK WELOPEN rows are ignored when no grid is supplied",
        well.cstatus is None,
    )


def validate_compositional_active_wi_masks_inactive_completions() -> int:
    model = OverallCompositionModel.__new__(OverallCompositionModel)
    well = _well("P4")
    well.cstatus = np.array([True, False, True])
    active_wi = model._well_active_wi(well)

    failures = 0
    failures += _check("compositional active WI zeros shut completions",
                       np.allclose(active_wi, [10.0, 0.0, 30.0]))
    failures += _check("compositional active WI does not mutate stored WI",
                       np.allclose(well.WI, [10.0, 20.0, 30.0]))
    return failures


def main() -> int:
    print("\nPhase 165: MRST-style completion status (`cstatus`) support")
    failures = validate_welopen_completion_range_sets_cstatus()
    failures += validate_welopen_completion_reopen_preserves_wi()
    failures += validate_positive_ijk_is_conservative_noop_without_grid_lookup()
    failures += validate_compositional_active_wi_masks_inactive_completions()
    if failures:
        print(f"\nPhase 165 failed ({failures} checks).")
        return 1
    print("\nCompletion-level status is represented and respected by compositional well WI.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
