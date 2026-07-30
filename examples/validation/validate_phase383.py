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
    s, gix, cross_section = slice_grid(
        g, np.array([[1.5, -1.0, 0.0], [1.5, 3.0, 0.0]]),
        cut_dir=np.array([0.0, 0.0, 1.0]), return_mapping=True,
        return_slice=True)
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
    failed += _check(
        "slice-grid cell count",
        cross_section.cells.num == int(ref["slice_cells_num"].item()),
    )
    failed += _check(
        "slice-grid face count",
        cross_section.faces.num == int(ref["slice_faces_num"].item()),
    )
    coords = np.asarray(cross_section.nodes.coords)
    coords = coords[np.lexsort(tuple(coords[:, i] for i in range(2, -1, -1)))]
    failed += _check(
        "slice-grid node coordinates",
        np.allclose(coords, ref["slice_coords"], rtol=0, atol=1e-14),
    )
    neighbor_pairs = np.sort(cross_section.faces.neighbors + 1, axis=1)
    neighbor_pairs = neighbor_pairs[np.lexsort(
        (neighbor_pairs[:, 1], neighbor_pairs[:, 0])
    )]
    failed += _check(
        "slice-grid neighbor-pair multiset",
        np.array_equal(neighbor_pairs, ref["slice_neighbor_pairs"]),
    )

    face_grid = compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
    face_grid, face_ix = slice_grid(
        face_grid, np.array([0.5, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.0]),
        only_faces=True, return_mapping=True,
    )
    failed += _check(
        "onlyFaces retains cell count",
        face_grid.cells.num == int(ref["only_cells_num"].item()),
    )
    failed += _check(
        "onlyFaces face count",
        face_grid.faces.num == int(ref["only_faces_num"].item()),
    )
    only_coords = np.asarray(face_grid.nodes.coords)
    only_coords = only_coords[np.lexsort(
        (only_coords[:, 2], only_coords[:, 1], only_coords[:, 0])
    )]
    failed += _check(
        "onlyFaces node coordinates",
        np.allclose(only_coords, ref["only_coords"], rtol=0, atol=1e-14),
    )
    failed += _check(
        "onlyFaces cell status counts",
        np.array_equal(
            np.bincount(face_ix.new.cells, minlength=4)[1:4],
            ref["only_new_cell_counts"].ravel(),
        ),
    )
    failed += _check(
        "onlyFaces face status counts",
        np.array_equal(
            np.bincount(face_ix.new.faces, minlength=4)[1:4],
            ref["only_new_face_counts"].ravel(),
        ),
    )
    only_parents = np.where(face_ix.parent.faces >= 0,
                            face_ix.parent.faces + 1, 0)
    only_parent_status = np.column_stack([only_parents, face_ix.new.faces])
    only_parent_status = only_parent_status[np.lexsort(
        (only_parent_status[:, 1], only_parent_status[:, 0])
    )]
    failed += _check(
        "onlyFaces parent/status multiset",
        np.array_equal(only_parent_status, ref["only_parent_face_status"]),
    )
    # MRST getIndices uses `cutFaceIx < nf_orig`, accidentally leaving the
    # final original face true when that face is split. PyMRST uses the
    # logically correct <= behavior, so every split parent is false here.
    split_parents = np.unique(face_ix.parent.faces[face_ix.new.faces == 2])
    failed += _check(
        "onlyFaces old-face mask corrects MRST final-face off-by-one",
        not np.any(face_ix.old.faces[split_parents]),
    )

    print("\nsliceGrid cell-splitting matches MRST!" if failed == 0
          else f"\n{failed} checks FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
