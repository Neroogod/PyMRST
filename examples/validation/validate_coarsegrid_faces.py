"""Validate the coarse-grid face topology against MRST.

Reference: ``dump_coarsegrid_faces_reference.m`` (MATLAB R2021b, MRST 2024a).

Five cases:

* **A** 3D, default indicator. The half-face tag on exterior faces is part of
  MRST's indicator, so a block's boundary is one coarse face per direction.
* **B** the same in 2D.
* **C** `showCoarseGrid.m`'s second example: a diagonal stripe indicator, run
  through `processFacePartition`.
* **D** a 3D coarse grid whose face partition comes from a finer partition.
* **E** `processFacePartition` on its own.

**What is compared, and what is not.** Cases A, B and D are compared array for
array, index for index. For anything involving `processFacePartition` the
*labels* are not comparable: MATLAB finds the connected components with
`dmperm` and this port uses `scipy.sparse.csgraph`, and neither promises a
particular numbering. The grouping is what carries meaning, so that is what is
checked, as a set of (block pair, fine-face set) tuples. A label comparison
would fail on a correct port, and a laxer check such as counting groups would
pass on a wrong one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io as sio

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.remove_cells import remove_cells
from pymrst.modules.coarsegrid.cell_partition_to_face_partition import (
    cell_partition_to_face_partition,
)
from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
from pymrst.modules.coarsegrid.partition_ui import partition_ui
from pymrst.modules.coarsegrid.process_face_partition import process_face_partition

REFERENCE = Path(__file__).with_name("ref_coarsegrid_faces.mat")


def load_reference():
    if not REFERENCE.exists():
        raise SystemExit(
            f"{REFERENCE.name} is missing. Regenerate it with\n"
            f"  matlab -batch \"startup; run('{REFERENCE.with_name('dump_coarsegrid_faces_reference.m')}')\"")
    return sio.loadmat(REFERENCE, squeeze_me=True, struct_as_record=False)


def compare_arrays(name, cg, ref) -> list[str]:
    """Every topology array, one-based on the MATLAB side."""
    checks = [
        ("faces.num", np.array([cg.num_faces]), np.array([int(ref.faces_num)])),
        ("faces.neighbors", cg.neighbors + 1, np.asarray(ref.faces_neighbors)),
        ("faces.connPos", cg.conn_pos + 1, np.asarray(ref.faces_connPos)),
        ("faces.fconn", cg.fconn + 1, np.asarray(ref.faces_fconn)),
        ("cells.facePos", cg.face_pos + 1, np.asarray(ref.cells_facePos)),
        ("cells.faces", cg.faces + 1, np.asarray(ref.cells_faces)[:, 0]),
        ("cells.faces tag", cg.face_indicator[cg.faces, 0],
         np.asarray(ref.cells_faces)[:, 1]),
    ]
    failures = []
    for field, mine, theirs in checks:
        if not np.array_equal(np.asarray(mine).reshape(np.asarray(theirs).shape),
                              np.asarray(theirs)):
            failures.append(f"{name}: {field} differs")
    return failures


def grouping(cg) -> set:
    """The coarse faces as (block pair, fine faces), free of any numbering."""
    return {(tuple(int(b) for b in np.sort(cg.neighbors[i])),
             frozenset(int(x) for x in cg.sub_faces(i)))
            for i in range(cg.num_faces)}


def matlab_grouping(ref) -> set:
    neighbors = np.asarray(ref.faces_neighbors, dtype=np.int64)
    conn_pos = np.asarray(ref.faces_connPos, dtype=np.int64) - 1
    fconn = np.asarray(ref.faces_fconn, dtype=np.int64) - 1
    return {(tuple(int(b) - 1 for b in np.sort(neighbors[i])),
             frozenset(int(x) for x in fconn[conn_pos[i]:conn_pos[i + 1]]))
            for i in range(int(ref.faces_num))}


def main() -> int:
    d = load_reference()
    failures: list[str] = []

    # ---------------------------------------------------------------- case A
    g = compute_geometry(cart_grid([6, 4, 2], [6.0, 4.0, 2.0]))
    p = partition_ui(g, [3, 2, 1])
    assert np.array_equal(p + 1, np.asarray(d["A"].p)), "partition A differs"
    cg = generate_coarse_grid(g, p)
    failures += compare_arrays("A", cg, d["A"])
    print(f"A  3D default indicator: {cg.num_faces} coarse faces "
          f"(MATLAB {int(d['A'].faces_num)})")

    # ---------------------------------------------------------------- case B
    g = compute_geometry(cart_grid([4, 4], [4.0, 4.0]))
    p = partition_ui(g, [2, 2])
    cg = generate_coarse_grid(g, p)
    failures += compare_arrays("B", cg, d["B"])
    print(f"B  2D default indicator: {cg.num_faces} coarse faces "
          f"(MATLAB {int(d['B'].faces_num)})")

    # ---------------------------------------------------------------- case C
    ref = d["C"]
    g = compute_geometry(cart_grid([8, 8], [1.0, 1.0]))
    c = np.asarray(g.cells.centroids, dtype=float)
    cell_tag = (np.sin(3.0 * np.pi * (c[:, 0] - c[:, 1])) > 0).astype(np.int64)
    assert np.array_equal(cell_tag + 1, np.asarray(ref.pf_cell)), "case C tag differs"
    pv = partition_ui(g, [2, 2])
    pf_in = cell_partition_to_face_partition(g, cell_tag)
    pf_out = process_face_partition(g, pv, pf_in)
    cg = generate_coarse_grid(g, pv, pf_out)

    # The face partition entering processFacePartition is fully determined, so
    # its grouping is comparable even though its labels are not.
    if not same_grouping(pf_in, np.asarray(ref.pf_in)):
        failures.append("C: cellPartitionToFacePartition grouping differs")
    if not same_grouping(pf_out, np.asarray(ref.pf_out)):
        failures.append("C: processFacePartition grouping differs")
    if grouping(cg) != matlab_grouping(ref):
        failures.append("C: coarse faces differ")
    print(f"C  2D stripe partition:  {cg.num_faces} coarse faces "
          f"(MATLAB {int(ref.faces_num)})")

    # ---------------------------------------------------------------- case D
    ref = d["D"]
    g = compute_geometry(cart_grid([20, 20, 6], [20.0, 20.0, 6.0]))
    cc = np.asarray(g.cells.centroids, dtype=float)
    g = compute_geometry(remove_cells(
        g, np.flatnonzero((cc[:, 0] < 10) & (cc[:, 1] < 10) & (cc[:, 2] < 3))))
    p = partition_ui(g, [2, 2, 2])
    q = partition_ui(g, [4, 4, 2])
    assert np.array_equal(p + 1, np.asarray(ref.p)), "partition D differs"
    assert np.array_equal(q + 1, np.asarray(ref.q)), "fine partition D differs"
    pf = cell_partition_to_face_partition(g, q)
    cg = generate_coarse_grid(g, p, pf)
    failures += compare_arrays("D", cg, ref)
    print(f"D  3D face partition:    {cg.num_faces} coarse faces "
          f"(MATLAB {int(ref.faces_num)})")

    # ---------------------------------------------------------------- case E
    ref = d["E"]
    g = compute_geometry(cart_grid([6, 6], [1.0, 1.0]))
    pv = partition_ui(g, [2, 2])
    cc = np.asarray(g.cells.centroids, dtype=float)
    cell_tag = ((cc[:, 0] < 0.5) == (cc[:, 1] < 0.5)).astype(np.int64)
    assert np.array_equal(cell_tag + 1, np.asarray(ref.pf_cell)), "case E tag differs"
    pf_in = cell_partition_to_face_partition(g, cell_tag)
    pf_out = process_face_partition(g, pv, pf_in)
    if not same_grouping(pf_out, np.asarray(ref.pf_out)):
        failures.append("E: processFacePartition grouping differs")
    print(f"E  processFacePartition: {len(set(pf_out.tolist()))} ids "
          f"(MATLAB {len(set(np.asarray(ref.pf_out).tolist()))})")

    print()
    if failures:
        for f in failures:
            print(f"FAIL {f}")
        return 1
    print("All coarse-grid face checks match MRST.")
    return 0


def same_grouping(mine, theirs) -> bool:
    """Do two labellings induce the same partition of the faces?

    Compares the equivalence relation, not the labels. Two labellings agree
    exactly when their joint labelling is no finer than either one alone, so
    counting distinct pairs against distinct labels settles it.
    """
    mine = np.asarray(mine).ravel()
    theirs = np.asarray(theirs).ravel()
    if mine.size != theirs.size:
        return False
    joint = np.unique(np.column_stack([mine, theirs]), axis=0).shape[0]
    return joint == np.unique(mine).size == np.unique(theirs).size


if __name__ == "__main__":
    raise SystemExit(main())
