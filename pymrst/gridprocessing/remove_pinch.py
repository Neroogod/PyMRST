"""Remove pinched-out layers from a grid (MRST `removePinch`).

Corner-point models often have layers that pinch out - cells of (near) zero
thickness where the top and bottom pillars meet. After processing, these leave
coincident nodes and degenerate faces/cells, and the layers above and below such
a pinch should be directly connected. This merges coincident nodes, drops the
collapsed faces (fewer than `griddim` distinct nodes) and cells (fewer than
`griddim + 1` faces), and then re-welds the faces that became coincident across
the pinch so the neighbouring cells communicate. The re-welding is exactly
`findMatchingFaces` + `removeInternalBoundary`.

MRST's optional canonical face-node reordering (rotate smallest node first, flip
for the second) is not reproduced: it exists only to let its `uniqueFaces` match
faces by node *sequence*, whereas `findMatchingFaces` matches by node *set* and
is order-independent, so the reconnected topology is the same without it.
"""

from typing import Optional

import numpy as np

from pymrst.gridprocessing.find_matching_faces import find_matching_faces
from pymrst.gridprocessing.private.remove_faces import remove_faces
from pymrst.gridprocessing.remove_cells import remove_cells
from pymrst.gridprocessing.remove_internal_boundary import remove_internal_boundary


def _dedupe_face_nodes(g) -> None:
    """Drop a node repeated within a face (adjacent or wrap-around), left over
    after coincident nodes were merged."""
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    new_nodes, counts = [], []
    for f in range(g.faces.num):
        seg = list(fnodes[node_pos[f]:node_pos[f + 1]])
        kept = [seg[i] for i in range(len(seg))
                if seg[i] != seg[(i - 1) % len(seg)]]      # drop consecutive dups
        # also collapse a face whose first and last coincide to a single run
        counts.append(len(kept))
        new_nodes += kept
    g.faces.nodes = np.asarray(new_nodes, dtype=np.int64)
    g.faces.nodePos = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)


def remove_pinch(g, tol: Optional[float] = None):
    """Return `g` with pinched layers removed and the surrounding cells
    reconnected."""
    coords = np.asarray(g.nodes.coords, dtype=float)

    # merge coincident nodes (exact), optionally near-coincident within tol
    uniq, inv = np.unique(coords, axis=0, return_inverse=True)
    g.nodes.coords = uniq
    g.nodes.num = uniq.shape[0]
    g.faces.nodes = inv.ravel()[np.asarray(g.faces.nodes, dtype=np.int64)]

    if tol is not None:
        d = np.concatenate([[np.inf],
                            np.sqrt(np.sum(np.diff(uniq, axis=0) ** 2, axis=1))])
        drop = d < tol
        keep = ~drop
        remap = np.cumsum(keep) - 1
        remap[drop] = remap[np.maximum(np.flatnonzero(keep)[
            np.searchsorted(np.flatnonzero(keep), np.flatnonzero(drop)) - 1], 0)]
        g.nodes.coords = uniq[keep]
        g.nodes.num = int(keep.sum())
        g.faces.nodes = remap[g.faces.nodes]

    _dedupe_face_nodes(g)

    pinched = np.flatnonzero(np.diff(np.asarray(g.faces.nodePos)) < g.griddim)
    if pinched.size:
        g, _ = remove_faces(g, pinched)

    degenerate = np.flatnonzero(
        np.diff(np.asarray(g.cells.facePos)) < g.griddim + 1)
    if degenerate.size:
        g = remove_cells(g, degenerate)

    N = find_matching_faces(g)                             # coincident faces
    if N.shape[0]:
        g, _ = remove_internal_boundary(g, N)              # reconnect across pinch
    return g


if __name__ == "__main__":
    from pymrst.gridprocessing.glue_2d_grid import _concatenate
    from pymrst.gridprocessing.grid_structure import Cells, Faces, Grid, Nodes
    from pymrst.gridprocessing.tessellation_grid import tessellation_grid

    def _duplicated_pair():
        """Two squares concatenated with duplicate nodes/faces on the shared
        x=1 edge - the situation removePinch is meant to clean up."""
        g1 = tessellation_grid(np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
                               [[0, 1, 2, 3]])
        g2 = tessellation_grid(np.array([[1., 0.], [2., 0.], [2., 1.], [1., 1.]]),
                               [[0, 1, 2, 3]])
        c = _concatenate(g1, g2)
        return Grid(cells=Cells(num=c.cells.num, facePos=c.cells.facePos,
                                faces=c.cells.faces,
                                indexMap=np.arange(c.cells.num)),
                    faces=Faces(num=c.faces.num, nodePos=c.faces.nodePos,
                                nodes=c.faces.nodes, neighbors=c.faces.neighbors),
                    nodes=Nodes(num=c.nodes.num, coords=c.nodes.coords),
                    griddim=2, type=["concat"])

    def test_pinch_reconnects_duplicated_grids():
        """After removePinch the two cells share one internal face; the 8
        duplicated nodes collapse to 6."""
        g = remove_pinch(_duplicated_pair())
        assert g.cells.num == 2, g.cells.num
        assert g.nodes.num == 6, g.nodes.num
        neigh = np.asarray(g.faces.neighbors)
        internal = np.all(neigh >= 0, axis=1)
        assert internal.sum() == 1 and set(neigh[internal][0]) == {0, 1}
        print(f"ok test_pinch_reconnects_duplicated_grids "
              f"({g.cells.num} cells, {g.faces.num} faces, {g.nodes.num} nodes)")

    def test_face_count_after_pinch():
        g = remove_pinch(_duplicated_pair())
        assert g.faces.num == 7, g.faces.num          # 8 - 1 merged pair
        print(f"ok test_face_count_after_pinch ({g.faces.num} faces)")

    def test_clean_grid_is_unchanged():
        """A grid with no coincident nodes passes through untouched."""
        g = tessellation_grid(np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]]),
                              [[0, 1, 2, 3]])
        nf, nn = g.faces.num, g.nodes.num
        g = remove_pinch(g)
        assert g.faces.num == nf and g.nodes.num == nn
        print("ok test_clean_grid_is_unchanged")

    test_pinch_reconnects_duplicated_grids()
    test_face_count_after_pinch()
    test_clean_grid_is_unchanged()
    print("\nAll tests passed!")
