"""Build a 2D grid from a set of 3D-grid faces (MRST `get2DGridFromFaces`).

Turns a chosen set of faces of a 3D grid into a standalone 2D grid: each 3D face
becomes a 2D cell (its polygon), and the polygon edges become the 2D faces, with
two neighbouring 3D faces that share an edge becoming adjacent 2D cells. It is the
surface/cross-section extractor `sliceGrid` builds on - and, more simply, the way
to get a mesh of a fault surface or a grid boundary for plotting or a 2D solve.
Shared edges are found with the same smaller-node-first canonical ordering used
by `tessellationGrid`.

MRST bug, corrected here: MRST's `get2DGridFromFaces` sets `g.nodes.num` to the
edge count `nf`, not the true node count - its own `g.nodes.coords` has only the
used-node rows. This port sets `nodes.num` consistent with the coordinates it
stores (verified against MRST's actual coords array in `validate_phase375.py`).
"""

import numpy as np

from pymrst.gridprocessing.grid_structure import Cells, Faces, Grid, Nodes


def _face_segments(g, f: np.ndarray):
    """Per selected face: its directed edges `(n1, n2)` and the face they came
    from (MRST `getFaceSegments`)."""
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    n1, n2, face_no = [], [], []
    for face in f:
        seg = fnodes[node_pos[face]:node_pos[face + 1]]
        m = seg.size
        for k in range(m):
            n1.append(seg[k])
            n2.append(seg[(k + 1) % m])
            face_no.append(face)
    return (np.asarray(n1, dtype=np.int64), np.asarray(n2, dtype=np.int64),
            np.asarray(face_no, dtype=np.int64))


def get_2d_grid_from_faces(g, f) -> Grid:
    """Return the 2D grid formed by the 3D faces `f` (indices or a boolean
    mask)."""
    f = np.asarray(f)
    if f.dtype == bool:
        f = np.flatnonzero(f)
    f = f.astype(np.int64)

    nix1, nix2, _ = _face_segments(g, f)
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    nn = (node_pos[f + 1] - node_pos[f]).astype(np.int64)     # edges per face

    # local node numbering over the nodes actually used
    used = np.unique(np.concatenate([nix1, nix2]))
    local = -np.ones(g.nodes.num, dtype=np.int64)
    local[used] = np.arange(used.size, dtype=np.int64)
    segs = np.column_stack([local[nix1], local[nix2]])

    cell_no = np.repeat(np.arange(1, f.size + 1), nn)         # 1-based 2D cell

    # canonical edges: smaller local node first; the side picks the neighbour col
    col = (segs[:, 0] > segs[:, 1]).astype(np.int64)
    canon = np.sort(segs, axis=1)
    edges, edgenum = np.unique(canon, axis=0, return_inverse=True)
    edgenum = edgenum.ravel()
    nf = edges.shape[0]

    neigh = np.zeros((nf, 2), dtype=np.int64)
    neigh[edgenum, col] = cell_no
    neigh = neigh - 1                                        # 0-based; outside -> -1

    coords = np.asarray(g.nodes.coords, dtype=float)[used]
    face_pos = np.concatenate([[0], np.cumsum(nn)]).astype(np.int64)
    node_pos2 = np.arange(0, 2 * nf + 1, 2, dtype=np.int64)

    g2 = Grid(
        cells=Cells(num=int(f.size), facePos=face_pos,
                    faces=edgenum.astype(np.int64),
                    indexMap=np.arange(f.size, dtype=np.int64)),
        faces=Faces(num=int(nf), nodePos=node_pos2,
                    nodes=edges.ravel().astype(np.int64), neighbors=neigh),
        nodes=Nodes(num=int(used.size), coords=coords),
        griddim=2,
        type=["get2DGridFromFaces"],
    )
    g2.cells.parent = f
    return g2


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_all_faces_of_one_cube():
        """The six faces of a single cube form a 2D grid of 6 cells, 12 unique
        edges and 8 nodes (the cube's own vertices/edges)."""
        g = compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
        g2 = get_2d_grid_from_faces(g, np.arange(g.faces.num))
        assert g2.cells.num == 6, g2.cells.num
        assert g2.nodes.num == 8, g2.nodes.num
        assert g2.faces.num == 12, g2.faces.num
        # each 2D cell (a cube face) has 4 edges
        assert np.all(np.diff(np.asarray(g2.cells.facePos)) == 4)
        print(f"ok test_all_faces_of_one_cube "
              f"({g2.cells.num} cells, {g2.faces.num} edges, {g2.nodes.num} nodes)")

    def test_a_single_face_becomes_a_boundary_only_cell():
        """One quad face alone -> one 2D cell with four boundary edges."""
        g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
        g2 = get_2d_grid_from_faces(g, np.array([0]))
        assert g2.cells.num == 1 and g2.faces.num == 4
        assert np.all(np.any(np.asarray(g2.faces.neighbors) == -1, axis=1))
        print("ok test_a_single_face_becomes_a_boundary_only_cell")

    def test_boolean_mask_selects_faces():
        g = compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
        mask = np.zeros(g.faces.num, dtype=bool)
        mask[:2] = True
        g2 = get_2d_grid_from_faces(g, mask)
        assert g2.cells.num == 2
        assert np.array_equal(g2.cells.parent, np.array([0, 1]))
        print("ok test_boolean_mask_selects_faces")

    def test_edges_reference_valid_cells():
        """Every 2D face's neighbours are valid 2D cell indices or -1, and the
        cells.faces indexing is consistent."""
        g = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
        g2 = get_2d_grid_from_faces(g, np.arange(6))
        neigh = np.asarray(g2.faces.neighbors)
        assert neigh.max() < g2.cells.num and neigh.min() >= -1
        assert np.asarray(g2.cells.faces).max() < g2.faces.num
        print("ok test_edges_reference_valid_cells")

    test_all_faces_of_one_cube()
    test_a_single_face_becomes_a_boundary_only_cell()
    test_boolean_mask_selects_faces()
    test_edges_reference_valid_cells()
    print("\nAll tests passed!")
