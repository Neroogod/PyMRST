"""Collapse grid edges shorter than a tolerance (MRST `removeShortEdges`).

Corner-point processing and grid gluing can leave slivers - edges of near-zero
length whose two nodes are effectively the same point. They make faces and cells
degenerate and the geometry ill-conditioned. This merges the endpoints of every
edge shorter than `tol`, then cleans up the fallout: faces that lost a node are
removed, cells left with too few faces are removed, and any faces that became
coincident are re-welded. It composes the internal-boundary and matching-face
tools built alongside it.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from pymrst.gridprocessing.find_matching_faces import find_matching_faces
from pymrst.gridprocessing.private.remove_faces import remove_faces
from pymrst.gridprocessing.remove_cells import remove_cells
from pymrst.gridprocessing.remove_internal_boundary import remove_internal_boundary


def _face_edges(g) -> np.ndarray:
    """Every face's directed edges `(n1, n2)` (cyclic)."""
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    e = []
    for f in range(g.faces.num):
        seg = fnodes[node_pos[f]:node_pos[f + 1]]
        m = seg.size
        for k in range(m):
            e.append((seg[k], seg[(k + 1) % m]))
    return np.asarray(e, dtype=np.int64) if e else np.zeros((0, 2), np.int64)


def _merge_nodes(g, edges: np.ndarray) -> None:
    """Merge the endpoints of `edges` into a single node each (averaged)."""
    if edges.size == 0:
        return
    n = g.nodes.num
    a = sp.csr_array((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                     shape=(n, n))
    a = a + a.T + sp.eye_array(n)
    _, labels = connected_components(a, directed=False)
    coords = np.asarray(g.nodes.coords, dtype=float)
    counts = np.bincount(labels)
    new = np.zeros((labels.max() + 1, coords.shape[1]))
    for d in range(coords.shape[1]):
        new[:, d] = np.bincount(labels, weights=coords[:, d]) / counts
    g.nodes.coords = new
    g.nodes.num = new.shape[0]
    g.faces.nodes = labels[np.asarray(g.faces.nodes, dtype=np.int64)]


def _remove_collapsed_faces(g):
    """Drop repeated nodes within each face, then faces with too few nodes."""
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    new_nodes, counts = [], []
    for f in range(g.faces.num):
        seg = fnodes[node_pos[f]:node_pos[f + 1]]
        seen, kept = set(), []
        for v in seg:
            if v not in seen:
                seen.add(v)
                kept.append(v)
        new_nodes += kept
        counts.append(len(kept))
    counts = np.asarray(counts, dtype=np.int64)
    g.faces.nodes = np.asarray(new_nodes, dtype=np.int64)
    g.faces.nodePos = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    collapsed = np.flatnonzero(counts < g.griddim)
    if collapsed.size:
        g, _ = remove_faces(g, collapsed)
    return g


def _remove_collapsed_cells(g):
    """Drop cells with too few faces, then re-weld any coincident faces."""
    nfaces = np.diff(np.asarray(g.cells.facePos, dtype=np.int64))
    collapsed = np.flatnonzero(nfaces < g.griddim + 1)
    if collapsed.size == 0:
        return g
    g = remove_cells(g, collapsed)
    N = find_matching_faces(g)
    if N.shape[0]:
        g, _ = remove_internal_boundary(g, N)
    return g


def remove_short_edges(g, tol: float = 0.0):
    """Return `g` with all edges shorter than `tol` collapsed."""
    edges = _face_edges(g)
    coords = np.asarray(g.nodes.coords, dtype=float)
    length = np.sqrt(((coords[edges[:, 0]] - coords[edges[:, 1]]) ** 2).sum(axis=1))
    _merge_nodes(g, edges[length < tol])
    g = _remove_collapsed_faces(g)
    g = _remove_collapsed_cells(g)
    return g


if __name__ == "__main__":
    from pymrst.gridprocessing.tessellation_grid import tessellation_grid

    def test_pentagon_with_a_sliver_becomes_a_quad():
        """A pentagon whose two top nodes are 1e-4 apart collapses to a quad when
        edges shorter than 1e-3 are removed."""
        p = np.array([[0., 0.], [1., 0.], [1., 1.], [0.5, 1.0001], [0.5, 1.]])
        g = tessellation_grid(p, [[0, 1, 2, 3, 4]])
        g = remove_short_edges(g, tol=1e-3)
        assert g.cells.num == 1, g.cells.num
        assert g.faces.num == 4, g.faces.num          # 5 edges -> 4 (sliver gone)
        assert g.nodes.num == 4, g.nodes.num
        print(f"ok test_pentagon_with_a_sliver_becomes_a_quad "
              f"({g.faces.num} faces, {g.nodes.num} nodes)")

    def test_no_short_edges_leaves_grid_unchanged():
        p = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
        g = tessellation_grid(p, [[0, 1, 2, 3]])
        nf, nn = g.faces.num, g.nodes.num
        g = remove_short_edges(g, tol=1e-6)
        assert g.faces.num == nf and g.nodes.num == nn
        print("ok test_no_short_edges_leaves_grid_unchanged")

    def test_merged_node_is_the_average():
        p = np.array([[0., 0.], [1., 0.], [1., 1.], [0.5, 1.0002], [0.5, 1.]])
        g = tessellation_grid(p, [[0, 1, 2, 3, 4]])
        g = remove_short_edges(g, tol=1e-3)
        # the merged top node sits at the average y of the two originals
        ys = np.asarray(g.nodes.coords)[:, 1]
        assert np.any(np.isclose(ys, 1.0001)), ys
        print("ok test_merged_node_is_the_average")

    test_pentagon_with_a_sliver_becomes_a_quad()
    test_no_short_edges_leaves_grid_unchanged()
    test_merged_node_is_the_average()
    print("\nAll tests passed!")
