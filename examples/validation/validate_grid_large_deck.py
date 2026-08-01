"""Validate large faulted corner-point decks against MRST.

The stress harness proves a big grid is internally consistent; this proves it
is the same grid MRST builds. Two tiers, so the reference stays small: full
sorted volumes and areas at 40x40x10, and counts plus a 21-point fingerprint
of the sorted distributions at 60x60x20. A third case pins the component split
of a deck cut into a lattice of islands.
"""

import warnings
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import compute_geometry
from pymrst.gridprocessing.process_grdecl import process_grdecl
from pymrst.gridprocessing.testgrids.one_sloping_fault import one_sloping_fault

REF = Path(__file__).with_name("ref_grid_large_deck.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def _sample(values: np.ndarray, count: int = 21) -> np.ndarray:
    """Sample sorted `values` at the same integer indices MATLAB used."""
    size = values.shape[0]
    index = (np.arange(count) * (size - 1)) // (count - 1)
    return np.sort(values)[index]


def _faulted(dims):
    grid = process_grdecl(one_sloping_fault(list(dims), drop=5.0))
    if isinstance(grid, list):
        raise AssertionError("faulted deck unexpectedly split into components")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return compute_geometry(grid, max_block_size=20_000)


def _close(actual, expected, tol=1e-9) -> tuple:
    actual = np.asarray(actual, dtype=float).ravel()
    expected = np.asarray(expected, dtype=float).ravel()
    if actual.shape != expected.shape:
        return False, f"shape {actual.shape} vs {expected.shape}"
    scale = max(float(np.abs(expected).max()), 1.0)
    error = float(np.abs(actual - expected).max()) / scale
    return error < tol, f"max relative |d| = {error:.3e}"


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_grid_large_deck_reference.m first")
        return 1
    ref = loadmat(REF, squeeze_me=True)
    failed = 0

    dims = tuple(int(d) for d in np.atleast_1d(ref["dims_full"]))
    print(f"faulted corner-point deck {dims}, full comparison:")
    grid = _faulted(dims)
    failed += _check(f"cell count ({grid.cells.num})",
                     grid.cells.num == int(ref["full_cells"]))
    failed += _check(f"face count ({grid.faces.num})",
                     grid.faces.num == int(ref["full_faces"]))
    failed += _check(f"node count ({grid.nodes.num})",
                     grid.nodes.num == int(ref["full_nodes"]))
    failed += _check("boundary face count",
                     int(np.sum(np.any(grid.faces.neighbors < 0, axis=1)))
                     == int(ref["full_boundary_faces"]))
    failed += _check("total volume",
                     *_close(grid.cells.volumes.sum(), ref["full_volume"]))
    failed += _check("sorted cell volumes",
                     *_close(np.sort(grid.cells.volumes),
                             ref["full_volumes_sorted"]))
    failed += _check("sorted face areas",
                     *_close(np.sort(grid.faces.areas),
                             ref["full_face_areas_sorted"]))

    dims = tuple(int(d) for d in np.atleast_1d(ref["dims_big"]))
    print(f"faulted corner-point deck {dims}, aggregates:")
    grid = _faulted(dims)
    failed += _check(f"cell count ({grid.cells.num})",
                     grid.cells.num == int(ref["big_cells"]))
    failed += _check(f"face count ({grid.faces.num})",
                     grid.faces.num == int(ref["big_faces"]))
    failed += _check(f"node count ({grid.nodes.num})",
                     grid.nodes.num == int(ref["big_nodes"]))
    failed += _check("boundary face count",
                     int(np.sum(np.any(grid.faces.neighbors < 0, axis=1)))
                     == int(ref["big_boundary_faces"]))
    failed += _check("total volume",
                     *_close(grid.cells.volumes.sum(), ref["big_volume"]))
    failed += _check("smallest cell volume",
                     *_close(grid.cells.volumes.min(), ref["big_min_volume"]))
    failed += _check("largest cell volume",
                     *_close(grid.cells.volumes.max(), ref["big_max_volume"]))
    failed += _check("cell volume distribution",
                     *_close(_sample(grid.cells.volumes),
                             ref["big_volume_quantiles"]))
    failed += _check("face area distribution",
                     *_close(_sample(grid.faces.areas),
                             ref["big_area_quantiles"]))

    dims = tuple(int(d) for d in np.atleast_1d(ref["dims_iso"]))
    print(f"deck cut into a lattice of islands {dims}:")
    nx, ny, nz = dims
    i, j, _ = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    active = (i % 3 != 2) & (j % 3 != 2)
    deck = one_sloping_fault([nx, ny, nz], drop=0.0)
    deck["ACTNUM"] = active.ravel(order="F").astype(np.int32)
    parts = process_grdecl(deck)
    parts = parts if isinstance(parts, list) else [parts]
    failed += _check(f"active cell count ({int(active.sum())})",
                     int(active.sum()) == int(ref["iso_active"]))
    failed += _check(f"component count ({len(parts)})",
                     len(parts) == int(ref["iso_components"]))
    failed += _check("component sizes",
                     np.array_equal(
                         np.sort([p.cells.num for p in parts]),
                         np.sort(np.atleast_1d(ref["iso_cells"]).astype(np.int64))))
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
