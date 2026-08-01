"""Check `make_planar_grid` against MRST (scripts/matlab/ref_make_planar_grid.mat).

Regenerate the reference from the MRST root with:

    matlab -batch "startup; run('.../scripts/matlab/dump_make_planar_grid_reference.m')"

MRST and PyMRST number the *parent's* faces differently, so the two runs select
a differently-numbered set of top faces and the surfaces they build come out in
a different order. Node numbering does agree - both `cartGrid`s lay nodes out
x-fastest - and the script asserts that first, via the coordinates.

Everything is then compared by node identity, which is invariant under the
face and cell numbering:

- the surface's edges, as a set of unordered node pairs;
- its cells, as a set of node sets;
- which cells each edge separates, as a set of (edge, cell-set) pairs - this is
  the part that would catch a wrong neighbour assignment.

The neighbour *columns* are not compared. They encode which way each cell winds
around its edge, and that follows from the parent's face node order, which is
itself numbering-dependent. The unordered pair is the claim that has to hold.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.io import loadmat

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymrst.gridprocessing import cart_grid, compute_geometry            # noqa: E402
from pymrst.utils.gridtools.make_planar_grid import make_planar_grid     # noqa: E402


def top_faces(h):
    centroids = np.asarray(h.faces.centroids, dtype=float)
    normals = np.asarray(h.faces.normals, dtype=float)
    flat = np.abs(normals[:, 2]) > np.abs(normals[:, :2]).sum(axis=1)
    return np.flatnonzero(flat & np.isclose(centroids[:, 2],
                                            centroids[:, 2].max()))


def describe(face_nodes, node_pos, neighbours, cell_faces, face_pos, num_cells):
    """Edges, cells and separations, all keyed by node identity."""
    edge = [frozenset(face_nodes[node_pos[f]:node_pos[f + 1]].tolist())
            for f in range(len(node_pos) - 1)]
    cell = [frozenset().union(*(edge[f] for f in
                                cell_faces[face_pos[c]:face_pos[c + 1]]))
            for c in range(num_cells)]
    separations = {(edge[f], frozenset(cell[c] for c in neighbours[f] if c >= 0))
                   for f in range(len(edge))}
    return set(edge), set(cell), separations


def main() -> int:
    reference = loadmat(Path(__file__).parent / "matlab" / "ref_make_planar_grid.mat",
                        squeeze_me=True, struct_as_record=False)
    ref, parent = reference["top"], reference["parent"]

    h = compute_geometry(cart_grid([3, 3, 2], [3.0, 3.0, 2.0]))
    aligned = np.allclose(np.asarray(h.nodes.coords, dtype=float),
                          np.asarray(parent.coords, dtype=float))
    print(f"  {'ok ' if aligned else 'FAIL'} parent node numbering agrees")
    if not aligned:
        print("\nFAILED: node numbering")
        return 1

    g = make_planar_grid(h, top_faces(h))
    failures = []
    for label, ok in (
        ("cell count", g.cells.num == int(ref.numCells)),
        ("face count", g.faces.num == int(ref.numFaces)),
        ("node count", g.nodes.num == int(ref.numNodes)),
    ):
        print(f"  {'ok ' if ok else 'FAIL'} {label}")
        if not ok:
            failures.append(label)

    mine = describe(np.asarray(g.faces.nodes, dtype=np.int64),
                    np.asarray(g.faces.nodePos, dtype=np.int64),
                    np.asarray(g.faces.neighbors, dtype=np.int64),
                    np.asarray(g.cells.faces, dtype=np.int64).ravel(),
                    np.asarray(g.cells.facePos, dtype=np.int64), g.cells.num)
    theirs = describe(np.asarray(ref.faceNodes, dtype=np.int64).ravel() - 1,
                      np.asarray(ref.nodePos, dtype=np.int64).ravel() - 1,
                      np.asarray(ref.neighbors, dtype=np.int64) - 1,
                      np.asarray(ref.cellFaces, dtype=np.int64).ravel() - 1,
                      np.asarray(ref.facePos, dtype=np.int64).ravel() - 1,
                      int(ref.numCells))

    for label, a, b in zip(("edges", "cells", "separations"), mine, theirs):
        ok = a == b
        print(f"  {'ok ' if ok else 'FAIL'} {label} ({len(a)} vs {len(b)})")
        if not ok:
            failures.append(label)

    print()
    if failures:
        print(f"FAILED: {', '.join(failures)}")
        return 1
    print("make_planar_grid matches MRST's makePlanarGrid.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
