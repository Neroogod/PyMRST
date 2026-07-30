"""Phase 383: sliceGrid cell-splitting against MRST (Phase 25).

Pins the PyMRST `slice_grid` clean-cut port against MRST's `sliceGrid` for a
vertical plane x=1.5 cutting a 3x2x2 grid (splitting the four middle-column
cells). Cell renumbering differs, so the check uses the cell count, the total
volume, and the sorted per-cell volumes - which pin the geometry of the split.

Only the clean planar cut (through cell interiors, no node snapping) is ported;
MRST's `topologicalCellSplit` handling of node-snapped / partial cuts is out of
scope.
"""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.slice_grid import slice_grid

REF = Path(__file__).with_name("ref_p383_slice.mat")


def _check(name: str, ok: bool, detail: str = "") -> int:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}{('  ' + detail) if detail else ''}")
    return 0 if ok else 1


def main() -> int:
    if not REF.exists():
        print(f"missing {REF.name}; run dump_phase383_reference.m in MATLAB first")
        return 1
    ref = loadmat(REF)

    g = compute_geometry(cart_grid([3, 2, 2], [3.0, 2.0, 2.0]))
    s, gix = slice_grid(
        g, np.array([[1.5, -1.0, 0.0], [1.5, 3.0, 0.0]]),
        cut_dir=np.array([0.0, 0.0, 1.0]), return_mapping=True)
    s = compute_geometry(s)

    failed = 0
    print("sliceGrid (3x2x2 cut at x=1.5) vs MRST:")
    failed += _check(f"cell count ({s.cells.num})",
                     s.cells.num == int(ref["cells_num"].ravel()[0]))
    failed += _check(f"total volume ({s.cells.volumes.sum():.4f})",
                     np.isclose(s.cells.volumes.sum(),
                                float(ref["total_vol"].ravel()[0])))
    vol = np.sort(np.asarray(s.cells.volumes))
    err = np.max(np.abs(vol - ref["vol_sorted"].ravel()))
    failed += _check("sorted per-cell volumes match", err < 1e-9,
                     f"max|dV| = {err:.3e}")
    failed += _check(
        "old cell count",
        np.count_nonzero(gix.old.cells) == int(ref["old_cell_count"].item()),
    )
    failed += _check(
        "old face count",
        np.count_nonzero(gix.old.faces) == int(ref["old_face_count"].item()),
    )
    cell_counts = np.bincount(gix.new.cells, minlength=4)[1:4]
    face_counts = np.bincount(gix.new.faces, minlength=4)[1:4]
    failed += _check(
        "new cell status counts",
        np.array_equal(cell_counts, ref["new_cell_status_counts"].ravel()),
    )
    failed += _check(
        "new face status counts",
        np.array_equal(face_counts, ref["new_face_status_counts"].ravel()),
    )
    failed += _check(
        "parent cells",
        np.array_equal(gix.parent.cells + 1, ref["parent_cells"].ravel()),
    )
    face_parent = np.where(gix.parent.faces >= 0, gix.parent.faces + 1, 0)
    face_parent_status = np.column_stack([face_parent, gix.new.faces])
    face_parent_status = face_parent_status[np.lexsort(
        (face_parent_status[:, 1], face_parent_status[:, 0])
    )]
    failed += _check(
        "face parent/status multiset",
        np.array_equal(face_parent_status, ref["face_parent_status"]),
    )

    print("\nsliceGrid cell-splitting matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
