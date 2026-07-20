"""Phase 166: grid-aware positive-I/J/K WELOPEN completion matching."""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pymrst.gridprocessing import cart_grid, compute_geometry  # noqa: E402
from pymrst.modules.compositional.facility import apply_deck_well_controls  # noqa: E402
from pymrst.params.wells_and_bc import Well  # noqa: E402


def _check(name: str, condition: bool) -> int:
    print(f"  {'ok  ' if condition else 'FAIL'} {name}")
    return 0 if condition else 1


def _grid():
    return compute_geometry(cart_grid([2, 2, 3], [20.0, 20.0, 30.0]))


def _well(name: str, cells: np.ndarray) -> Well:
    cells = np.asarray(cells, dtype=np.int64)
    return Well(
        cells=cells,
        type="bhp",
        val=1.0,
        r=np.ones(cells.size),
        dir="z",
        WI=np.linspace(10.0, 10.0 * cells.size, cells.size),
        dZ=np.zeros(cells.size),
        name=name,
        compi=np.asarray([0.5, 0.5], dtype=float),
        refDepth=0.0,
        lims={},
    )


def validate_positive_ijk_matches_cartesian_perforations() -> int:
    grid = _grid()
    # One well down deck column I=1, J=1. Active cell ids are K=1,2,3.
    well = _well("P1", np.array([0, 4, 8]))
    out = apply_deck_well_controls(
        [well],
        {
            "WCONPROD": [["P1", "OPEN", "BHP", np.inf, np.inf, np.inf, np.inf, np.inf, 100.0]],
            "WELOPEN": [["P1", "SHUT", 1, 1, 2, 1, 3]],
        },
        grid=grid,
    )[0]

    return _check(
        "positive-I/J/K WELOPEN shuts the matching Cartesian perforation",
        np.array_equal(out.cstatus, np.array([True, False, True])),
    )


def validate_zero_ijk_wildcards_with_perf_bounds() -> int:
    grid = _grid()
    well = _well("P2", np.array([1, 5, 9]))
    out = apply_deck_well_controls(
        [well],
        {
            "WCONPROD": [["P2", "OPEN", "BHP", np.inf, np.inf, np.inf, np.inf, np.inf, 100.0]],
            "WELOPEN": [["P2", "SHUT", 0, 0, 0, 2, 3]],
        },
        grid=grid,
    )[0]

    return _check(
        "zero I/J/K values act as MRST wildcards with completion bounds",
        np.array_equal(out.cstatus, np.array([True, False, False])),
    )


def validate_positive_ijk_without_grid_remains_noop() -> int:
    well = _well("P3", np.array([0, 4, 8]))
    out = apply_deck_well_controls(
        [well],
        {
            "WCONPROD": [["P3", "OPEN", "BHP", np.inf, np.inf, np.inf, np.inf, np.inf, 100.0]],
            "WELOPEN": [["P3", "SHUT", 1, 1, 2, 1, 3]],
        },
    )[0]

    return _check("positive-I/J/K WELOPEN remains conservative without grid", out.cstatus is None)


def validate_default_negative_ijk_does_not_act_as_wildcard() -> int:
    grid = _grid()
    well = _well("P4", np.array([0, 4, 8]))
    out = apply_deck_well_controls(
        [well],
        {
            "WCONPROD": [["P4", "OPEN", "BHP", np.inf, np.inf, np.inf, np.inf, np.inf, 100.0]],
            "WELOPEN": [["P4", "SHUT", 1, -1, -1, 1, 3]],
        },
        grid=grid,
    )[0]

    return _check(
        "default-negative partial I/J/K rows do not masquerade as wildcards",
        np.array_equal(out.cstatus, np.array([True, True, True])),
    )


def validate_active_index_map_is_used() -> int:
    grid = _grid()
    # Simulate an extracted active grid by keeping a nontrivial indexMap.
    grid.cells.indexMap = np.array([0, 4, 8, 1, 5, 9], dtype=np.int64)
    well = _well("P5", np.array([0, 1, 2]))
    out = apply_deck_well_controls(
        [well],
        {
            "WCONPROD": [["P5", "OPEN", "BHP", np.inf, np.inf, np.inf, np.inf, np.inf, 100.0]],
            "WELOPEN": [["P5", "SHUT", 1, 1, 3, 1, 3]],
        },
        grid=grid,
    )[0]

    return _check(
        "positive-I/J/K matching uses cells.indexMap for active grids",
        np.array_equal(out.cstatus, np.array([True, True, False])),
    )


def main() -> int:
    print("\nPhase 166: grid-aware positive-I/J/K WELOPEN completion matching")
    failures = validate_positive_ijk_matches_cartesian_perforations()
    failures += validate_zero_ijk_wildcards_with_perf_bounds()
    failures += validate_positive_ijk_without_grid_remains_noop()
    failures += validate_default_negative_ijk_does_not_act_as_wildcard()
    failures += validate_active_index_map_is_used()
    if failures:
        print(f"\nPhase 166 failed ({failures} checks).")
        return 1
    print("\nPositive-I/J/K WELOPEN rows now map to Cartesian well completions when a grid is supplied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
