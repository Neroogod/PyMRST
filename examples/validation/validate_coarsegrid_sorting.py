"""Validate `partition_tensor` and `sort_sub_edges` against MRST.

Reference: ``dump_coarsegrid_sorting_reference.m`` (MATLAB R2021b, MRST 2024a).

* **A** `partitionTensor` in 3D: uneven widths, even widths, and unit widths.
* **B** the same on a grid with cells removed, where a block can be cut in two.
* **C** the 2D form.
* **D** `sortSubEdges` on a 6x6 grid in 2x2 blocks: both the fine faces behind
  each coarse face and the coarse faces around each block.

`partitionTensor` labels its blocks in whatever order the fold happened to
produce, so the grouping is compared rather than the labels. `sortSubEdges`
produces an *order*, which is the whole point of it, so that is compared
element for element.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.remove_cells import remove_cells
from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
from pymrst.modules.coarsegrid.partition_cart_grid import partition_cart_grid
from pymrst.modules.coarsegrid.partition_tensor import partition_tensor
from pymrst.modules.coarsegrid.sort_sub_edges import sort_sub_edges

REFERENCE = Path(__file__).with_name("ref_coarsegrid_sorting.mat")


def load_reference():
    if not REFERENCE.exists():
        raise SystemExit(
            f"{REFERENCE.name} is missing. Regenerate it with\n"
            f"  matlab -batch \"startup; run('"
            f"{REFERENCE.with_name('dump_coarsegrid_sorting_reference.m')}')\"")
    return sio.loadmat(REFERENCE, squeeze_me=True, struct_as_record=False)


def same_grouping(mine, theirs) -> bool:
    mine, theirs = np.asarray(mine).ravel(), np.asarray(theirs).ravel()
    if mine.size != theirs.size:
        return False
    joint = np.unique(np.column_stack([mine, theirs]), axis=0).shape[0]
    return joint == np.unique(mine).size == np.unique(theirs).size


def main() -> int:
    d = load_reference()
    failures: list[str] = []

    # ---------------------------------------------------------------- case A
    ref = d["A"]
    g = compute_geometry(cart_grid([6, 4, 2], [6.0, 4.0, 2.0]))
    for label, args, key in (
            ("uneven", ([1, 4, 1], [4], [2]), "p_uneven"),
            ("even", ([3, 3], [2, 2], [2]), "p_even"),
            ("unit", ([1] * 6, [1] * 4, [1] * 2), "p_unit")):
        p = partition_tensor(g, *args)
        theirs = np.asarray(getattr(ref, key))
        if not same_grouping(p, theirs):
            failures.append(f"A: partitionTensor {label} grouping differs")
        if int(p.max()) + 1 != int(theirs.max()):
            failures.append(f"A: partitionTensor {label} block count differs")
    print(f"A  partitionTensor 3D: "
          f"{int(np.asarray(ref.p_uneven).max())}, "
          f"{int(np.asarray(ref.p_even).max())}, "
          f"{int(np.asarray(ref.p_unit).max())} blocks")

    # ---------------------------------------------------------------- case B
    ref = d["B"]
    g = compute_geometry(remove_cells(
        cart_grid([6, 4, 1], [6.0, 4.0, 1.0]), np.array([7, 8, 13, 14])))
    if not np.array_equal(np.asarray(g.cells.indexMap) + 1,
                          np.asarray(ref.indexMap)):
        failures.append("B: the grid with cells removed differs")
    p = partition_tensor(g, [3, 3], [2, 2], [1])
    if not same_grouping(p, np.asarray(ref.p)):
        failures.append("B: partitionTensor grouping differs on the holed grid")
    print(f"B  with a hole:        {int(p.max()) + 1} blocks "
          f"(MATLAB {int(np.asarray(ref.p).max())})")

    # ---------------------------------------------------------------- case C
    ref = d["C"]
    g = compute_geometry(cart_grid([6, 4], [6.0, 4.0]))
    p = partition_tensor(g, [2, 4], [1, 3])
    if not same_grouping(p, np.asarray(ref.p)):
        failures.append("C: partitionTensor 2D grouping differs")
    print(f"C  partitionTensor 2D: {int(p.max()) + 1} blocks "
          f"(MATLAB {int(np.asarray(ref.p).max())})")

    # ---------------------------------------------------------------- case D
    ref = d["D"]
    g = compute_geometry(cart_grid([6, 6], [6.0, 6.0]))
    p = partition_cart_grid([6, 6, 1], [2, 2, 1])
    assert np.array_equal(p + 1, np.asarray(ref.p)), "partition D differs"
    cg = generate_coarse_grid(g, p)

    for field, mine, theirs in (
            ("connPos", cg.conn_pos + 1, ref.connPos),
            ("facePos", cg.face_pos + 1, ref.facePos),
            ("neighbors", cg.neighbors + 1, ref.neighbors),
            ("fconn before sorting", cg.fconn + 1, ref.fconn_before),
            ("cells.faces before sorting", cg.faces + 1, ref.faces_before)):
        theirs = np.asarray(theirs)
        if not np.array_equal(np.asarray(mine).reshape(theirs.shape), theirs):
            failures.append(f"D: {field} differs")

    sort_sub_edges(cg)
    for field, mine, theirs in (
            ("fconn after sorting", cg.fconn + 1, ref.fconn_after),
            ("cells.faces after sorting", cg.faces + 1, ref.faces_after)):
        theirs = np.asarray(theirs)
        if not np.array_equal(np.asarray(mine).reshape(theirs.shape), theirs):
            failures.append(f"D: {field} differs")

    # The comparison only means something if sorting changed something.
    if np.array_equal(np.asarray(ref.fconn_before), np.asarray(ref.fconn_after)):
        failures.append("D: sortSubEdges reordered nothing, so nothing is tested")
    print(f"D  sortSubEdges:       {cg.num_faces} coarse faces, "
          f"{cg.fconn.size} sub-edges reordered")

    print()
    if failures:
        for f in failures:
            print(f"FAIL {f}")
        return 1
    print("All coarse-grid sorting checks match MRST.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
