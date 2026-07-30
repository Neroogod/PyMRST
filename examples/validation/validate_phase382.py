"""Phase 382: computeGridSlicePolygons core against MRST (Phase 25).

Pins the PyMRST `compute_grid_slice_polygons` core against MRST's
`computeGridSlicePolygons` for a vertical plane sweeping a diagonal segment
across a 3x3x2 grid. The set of cut cells (with their polygon vertex counts) and
the sorted set of 3D intersection coordinates are compared.

This is the geometric core of `sliceGrid`; the grid *cutting* (splitFaces /
splitCells) that consumes these polygons is a separate, larger port and is not
covered here.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.compute_grid_slice_polygons import (
    compute_grid_slice_polygons,
)

REF = Path(__file__).with_name("ref_p382_slicepoly.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase382_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)
    cell_ref = ref["cell_ix"].ravel().astype(np.int64) - 1     # 1- -> 0-based
    nvert_ref = ref["nvert"].ravel().astype(np.int64)

    g = compute_geometry(cart_grid([3, 3, 2], [3.0, 3.0, 2.0]))
    poly = compute_grid_slice_polygons(
        g, np.array([[1.5, -0.5, 0.0], [1.5, 3.5, 0.0]]),
        cut_dir=np.array([0.0, 0.0, 1.0]))
    cells = poly["cell_ix"]
    nvert = np.diff(poly["node_pos"])

    failed = 0
    print("computeGridSlicePolygons (diagonal vertical cut, 3x3x2) vs MRST:")
    # compare the (cell, vertex-count) sets, order-independent
    mine = set(zip(cells.tolist(), nvert.tolist()))
    theirs = set(zip(cell_ref.tolist(), nvert_ref.tolist()))
    failed += _check(f"same cut cells and vertex counts "
                     f"({cells.size} cells)", mine == theirs)

    coords = np.round(poly["coords_3d"], 6)
    coords = coords[np.lexsort(coords.T[::-1])]
    ref_coords = np.asarray(ref["coords3d"], dtype=float)
    err = (np.max(np.abs(coords - ref_coords))
           if coords.shape == ref_coords.shape else np.inf)
    failed += _check("all intersection coordinates match", err < 1e-6,
                     f"max|dx| = {err:.3e}")

    gm = compute_geometry(cart_grid([3, 3, 1], [3.0, 3.0, 1.0]))
    multi = compute_grid_slice_polygons(
        gm, np.array([[.5, 0, 0], [.5, 1.5, 0], [2.5, 1.5, 0]]),
        cut_dir=np.array([0.0, 0.0, 1.0]),
    )
    multi_pairs = sorted(zip(
        multi["cell_ix"].tolist(), np.diff(multi["node_pos"]).tolist()
    ))
    multi_ref_pairs = sorted(zip(
        (ref["multi_cell_ix"].ravel().astype(int) - 1).tolist(),
        ref["multi_nvert"].ravel().astype(int).tolist(),
    ))
    failed += _check(
        "multi-segment cells and vertex counts",
        multi_pairs == multi_ref_pairs,
    )
    multi_coords = np.round(multi["coords_3d"], 6)
    multi_coords = multi_coords[np.lexsort(multi_coords.T[::-1])]
    multi_ref_coords = np.asarray(ref["multi_coords3d"], dtype=float)
    multi_err = (
        np.max(np.abs(multi_coords - multi_ref_coords))
        if multi_coords.shape == multi_ref_coords.shape else np.inf
    )
    failed += _check(
        "multi-segment coordinates", multi_err < 1e-6,
        f"max|dx| = {multi_err:.3e}",
    )

    gr = compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
    bounded = compute_grid_slice_polygons(
        gr, np.array([[.5, 0, .5], [.5, 1, .5]]),
        cut_dir=np.array([0.0, 0.0, 1.0]), radius=.2,
    )
    radius_coords = np.round(bounded["coords_3d"], 6)
    radius_coords = radius_coords[np.lexsort(radius_coords.T[::-1])]
    radius_ref_coords = np.asarray(ref["radius_coords3d"], dtype=float)
    radius_err = (
        np.max(np.abs(radius_coords - radius_ref_coords))
        if radius_coords.shape == radius_ref_coords.shape else np.inf
    )
    failed += _check(
        "bounded-radius coordinates", radius_err < 1e-6,
        f"max|dx| = {radius_err:.3e}",
    )

    print("\ncomputeGridSlicePolygons core matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
