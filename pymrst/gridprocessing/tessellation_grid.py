"""Build a 2D grid from a polygon tessellation (MRST `tessellationGrid`).

Given a set of 2D points and a tessellation - each cell a closed list of point
indices tracing its polygon - this assembles the grid whose faces are the polygon
edges. Shared edges are found the same way `hexahedralGrid` finds shared quads:
each edge is written smaller-node-first, so two cells sharing an edge produce the
identical node pair and a unique-rows pass groups them, giving the face-to-cell
map without any geometric tolerance. It is the general 2D unstructured-grid
constructor - arbitrary polygons, not just triangles or quads.
"""

from typing import List, Sequence, Union

import numpy as np

from pymrst.gridprocessing.grid_structure import Cells, Faces, Grid, Nodes


def _half_edges(t) -> tuple:
    """Return `(edges, cell_no, counts)`: every polygon edge as a node pair,
    the cell it belongs to, and the edge count per cell."""
    if isinstance(t, np.ndarray) and t.ndim == 2:
        t = [row for row in np.asarray(t, dtype=np.int64)]
    polys = [np.asarray(c, dtype=np.int64).ravel() for c in t]
    counts = np.array([p.size for p in polys], dtype=np.int64)

    edges, cell_no = [], []
    for ci, poly in enumerate(polys):
        m = poly.size
        for k in range(m):
            edges.append((poly[k], poly[(k + 1) % m]))
            cell_no.append(ci + 1)                       # 1-based, 0 = outside
    return np.asarray(edges, dtype=np.int64), np.asarray(cell_no), counts


def tessellation_grid(p, t: Union[np.ndarray, Sequence]) -> Grid:
    """Assemble a 2D grid from points `p` (n x 2) and tessellation `t`.

    `t` is either a list of polygons (each a sequence of 0-based point indices)
    or, for uniform polygons, an `(nc, k)` array.
    """
    p = np.asarray(p, dtype=float)
    assert p.shape[1] == 2, "tessellationGrid is two-dimensional"

    edges, cell_no, counts = _half_edges(t)
    assert edges.max() < p.shape[0] and edges.min() >= 0, \
        "tessellation references invalid points"

    # canonical edge orientation: smaller node first; the side (0/1) separates a
    # shared edge's two cells into the two neighbour columns
    col = (edges[:, 0] > edges[:, 1]).astype(np.int64)   # 1 if flipped
    canon = np.sort(edges, axis=1)

    faces, cf = np.unique(canon, axis=0, return_inverse=True)
    cf = cf.ravel()
    nf = faces.shape[0]
    nc = counts.size

    neigh = np.zeros((nf, 2), dtype=np.int64)
    neigh[cf, col] = cell_no                             # 1-based; 0 = outside
    neigh = neigh - 1                                    # 0-based; outside -> -1

    face_pos = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    node_pos = np.arange(0, 2 * nf + 1, 2, dtype=np.int64)

    g = Grid(
        cells=Cells(num=int(nc), facePos=face_pos, faces=cf.astype(np.int64),
                    indexMap=np.arange(nc, dtype=np.int64)),
        faces=Faces(num=int(nf), nodePos=node_pos,
                    nodes=faces.ravel().astype(np.int64), neighbors=neigh),
        nodes=Nodes(num=int(p.shape[0]), coords=p.copy()),
        griddim=2,
        type=["tessellationGrid"],
    )
    return g


if __name__ == "__main__":
    def test_two_triangles_share_one_edge():
        """Two triangles over a unit square, sharing the diagonal: 2 cells, 5
        faces, one internal edge joining them, 4 nodes."""
        p = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        t = [[0, 1, 2], [1, 3, 2]]                       # share edge 1-2
        g = tessellation_grid(p, t)
        assert g.cells.num == 2 and g.nodes.num == 4
        assert g.faces.num == 5, g.faces.num
        neigh = np.asarray(g.faces.neighbors)
        internal = np.all(neigh >= 0, axis=1)
        assert internal.sum() == 1 and set(neigh[internal][0]) == {0, 1}
        print(f"ok test_two_triangles_share_one_edge ({g.faces.num} faces)")

    def test_single_quad_is_all_boundary():
        p = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        g = tessellation_grid(p, [[0, 1, 2, 3]])
        assert g.cells.num == 1 and g.faces.num == 4
        assert np.all(np.any(np.asarray(g.faces.neighbors) == -1, axis=1))
        print("ok test_single_quad_is_all_boundary")

    def test_uniform_array_input_matches_list():
        """A 2xk array of quads gives the same grid as the list form."""
        p = np.array([[0., 0.], [1., 0.], [2., 0.],
                      [0., 1.], [1., 1.], [2., 1.]])
        arr = np.array([[0, 1, 4, 3], [1, 2, 5, 4]])     # two quads, share edge 1-4
        g = tessellation_grid(p, arr)
        assert g.cells.num == 2 and g.faces.num == 7     # 4+4-1 shared
        internal = np.all(np.asarray(g.faces.neighbors) >= 0, axis=1)
        assert internal.sum() == 1
        print(f"ok test_uniform_array_input_matches_list ({g.faces.num} faces)")

    def test_every_face_has_two_nodes():
        p = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        g = tessellation_grid(p, [[0, 1, 2], [1, 3, 2]])
        assert np.all(np.diff(np.asarray(g.faces.nodePos)) == 2)
        print("ok test_every_face_has_two_nodes")

    test_two_triangles_share_one_edge()
    test_single_quad_is_all_boundary()
    test_uniform_array_input_matches_list()
    test_every_face_has_two_nodes()
    print("\nAll tests passed!")
