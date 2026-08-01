"""Augmented VEM grid geometry (MRST `computeVEMGeometry` / `createAugmentedGrid`).

Adds the mappings the virtual element solver needs beyond `compute_geometry`:
cell (and, in 3-D, face) diameters, and - in 3-D - the full edge topology that
2-D faces already carry implicitly. Edges are built from the cyclic node pairs of
each face and deduplicated, giving `G.edges`, per-face `edges/edgePos/edgeSign`,
in-plane `faces.edgeNormals`, and `cells.edges/edgePos`. This unblocks 3-D VEM
inner products (which project onto each face's 2-D VEM space).
"""

import numpy as np
import scipy.sparse as sp

from pymrst.gridprocessing.grid_structure import Edges


def _diameters(coords: np.ndarray, pos: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Max pairwise node distance for each cell/face (its geometric diameter)."""
    n = pos.size - 1
    out = np.zeros(n)
    for i in range(n):
        pts = coords[ids[pos[i]:pos[i + 1]]]
        d = pts[:, None, :] - pts[None, :, :]
        out[i] = np.sqrt((d ** 2).sum(axis=2)).max()
    return out


def _build_edges(g):
    """Deduplicate cyclic face-node pairs into a global edge list (3-D).

    Returns `(edges_nodes, faces_edges, faces_edge_sign)` where `edges_nodes` is
    the `(ne, 2)` canonical (ascending) node pairs, `faces_edges` maps each
    face-edge to its global edge, and `faces_edge_sign` is +1 when the face
    traverses the edge in canonical direction, -1 otherwise.
    """
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)

    starts, nexts = [], []
    for f in range(g.faces.num):
        fn = fnodes[node_pos[f]:node_pos[f + 1]]
        starts.append(fn)
        nexts.append(np.roll(fn, -1))              # cyclic next node
    eh = np.column_stack([np.concatenate(starts), np.concatenate(nexts)])
    ehs = np.sort(eh, axis=1)                       # undirected canonical form
    edges_nodes, j = np.unique(ehs, axis=0, return_inverse=True)
    j = j.astype(np.int64)
    sign = np.where(edges_nodes[j, 0] == eh[:, 0], 1, -1).astype(np.int64)
    return edges_nodes, j, sign


def compute_vem_geometry(g):
    """Augment `g` in place with VEM geometry and return it.

    2-D adds `cells.diameters`. 3-D additionally builds `g.edges`
    (nodes/nodePos/lengths/centroids), `faces.edges/edgePos/edgeSign`,
    `faces.edgeNormals`, `cells.edges/edgePos`, and `faces.diameters`.
    """
    coords = np.asarray(g.nodes.coords, dtype=float)
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)

    if g.cells.nodes is None:
        _augment_cell_nodes(g)
    g.cells.diameters = _diameters(coords, np.asarray(g.cells.nodePos, np.int64),
                                   np.asarray(g.cells.nodes, np.int64))
    if "computeVEMGeometry" not in g.type:
        g.type.append("computeVEMGeometry")
    if g.griddim == 2:
        return g

    # --- Edge topology (3-D).
    edges_nodes, faces_edges, edge_sign = _build_edges(g)
    ne = edges_nodes.shape[0]
    nfe = np.diff(node_pos)                          # edges per face = nodes/face
    face_edge_pos = np.concatenate([[0], np.cumsum(nfe)])

    lengths = np.linalg.norm(coords[edges_nodes[:, 1]] - coords[edges_nodes[:, 0]],
                             axis=1)
    centroids = 0.5 * (coords[edges_nodes[:, 0]] + coords[edges_nodes[:, 1]])
    g.edges = Edges(num=int(ne),
                    nodes=edges_nodes.ravel(),
                    nodePos=np.arange(0, 2 * ne + 1, 2, dtype=np.int64),
                    lengths=lengths, centroids=centroids)

    # --- In-plane edge normals, one per face-edge (cross(edge, face-normal)).
    face_of_edge = np.repeat(np.arange(g.faces.num), nfe)
    en = edges_nodes[faces_edges]                   # (nfe_total, 2)
    evec = (coords[en[:, 1]] - coords[en[:, 0]]) * edge_sign[:, None]
    fnrm = np.asarray(g.faces.normals, dtype=float)[face_of_edge]
    normals = np.cross(evec, fnrm)
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    g.faces.edges = faces_edges
    g.faces.edgePos = face_edge_pos
    g.faces.edgeSign = edge_sign
    g.faces.edgeNormals = normals

    # --- Cell edges: an edge belongs to a cell iff both its nodes are cell nodes.
    E = sp.csr_array((np.ones(2 * ne), (np.repeat(np.arange(ne), 2),
                                        edges_nodes.ravel())),
                     shape=(ne, g.nodes.num))
    cell_nodes = np.asarray(g.cells.nodes, dtype=np.int64)
    cnp = np.asarray(g.cells.nodePos, dtype=np.int64)
    cell_of_node = np.repeat(np.arange(g.cells.num), np.diff(cnp))
    C = sp.csr_array((np.ones(cell_nodes.size), (cell_nodes, cell_of_node)),
                     shape=(g.nodes.num, g.cells.num))
    inc = (E @ C).tocoo()
    hit = inc.data == 2
    ce, cc = inc.row[hit], inc.col[hit]
    order = np.lexsort((ce, cc))                     # sort by cell, then edge
    g.cells.edges = ce[order].astype(np.int64)
    g.cells.edgePos = np.concatenate(
        [[0], np.cumsum(np.bincount(cc, minlength=g.cells.num))]).astype(np.int64)

    g.faces.diameters = _diameters(coords, node_pos, fnodes)
    return g


def _augment_cell_nodes(g):
    """Build `cells.nodes/nodePos` (unique cell nodes) if not already present."""
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
    nodes, pos = [], [0]
    for c in range(g.cells.num):
        faces = cell_faces[face_pos[c]:face_pos[c + 1]]
        cn = np.unique(np.concatenate(
            [fnodes[node_pos[f]:node_pos[f + 1]] for f in faces]))
        nodes.append(cn)
        pos.append(pos[-1] + cn.size)
    g.cells.nodes = np.concatenate(nodes).astype(np.int64)
    g.cells.nodePos = np.asarray(pos, dtype=np.int64)


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_2d_only_diameters():
        g = compute_vem_geometry(compute_geometry(cart_grid([3, 3], [3.0, 3.0])))
        assert g.cells.diameters.shape == (9,)
        # Unit square cell diameter is its diagonal sqrt(2).
        assert np.allclose(g.cells.diameters, np.sqrt(2.0))
        assert g.edges is None
        print("ok test_2d_only_diameters")

    def test_3d_edge_topology_cube():
        g = compute_vem_geometry(compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0])))
        # A single cube: 8 nodes, 12 edges, each of length 1.
        assert g.edges.num == 12, g.edges.num
        assert np.allclose(g.edges.lengths, 1.0)
        assert g.cells.edges.size == 12 and g.cells.edgePos[-1] == 12
        # Every face is a unit square: 4 edges, diameter = diagonal sqrt(2).
        assert np.all(np.diff(g.faces.edgePos) == 4)
        assert np.allclose(g.faces.diameters, np.sqrt(2.0))
        assert np.allclose(g.cells.diameters, np.sqrt(3.0))    # cube diagonal
        print("ok test_3d_edge_topology_cube")

    def test_3d_edge_normals_are_in_plane_and_unit():
        g = compute_vem_geometry(compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0])))
        en = g.faces.edgeNormals
        assert np.allclose(np.linalg.norm(en, axis=1), 1.0)
        # Edge normal lies in the face plane: orthogonal to the face normal.
        face_of_edge = np.repeat(np.arange(g.faces.num), np.diff(g.faces.edgePos))
        fn = g.faces.normals[face_of_edge]
        fn = fn / np.linalg.norm(fn, axis=1, keepdims=True)
        assert np.max(np.abs((en * fn).sum(axis=1))) < 1e-12
        print("ok test_3d_edge_normals_are_in_plane_and_unit")

    def test_edges_shared_between_faces_are_unique():
        g = compute_vem_geometry(compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0])))
        # The shared interior face's 4 edges must reuse global edge ids.
        assert g.edges.num < g.faces.edges.size          # dedup happened
        assert g.edges.nodes.size == 2 * g.edges.num
        print("ok test_edges_shared_between_faces_are_unique")

    test_2d_only_diameters()
    test_3d_edge_topology_cube()
    test_3d_edge_normals_are_in_plane_and_unit()
    test_edges_shared_between_faces_are_unique()
    print("\nAll tests passed!")
