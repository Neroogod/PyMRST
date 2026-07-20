"""Grid from points and a triangle list, ported from MRST triangleGrid.m.

Each triangle becomes a cell with three edge faces. Shared edges are found
by sorting each edge's two nodes and uniquifying, and the orientation of
the half-edge (ascending or descending node pair) decides which neighbor
column the cell lands in, exactly as in MATLAB.
"""

from typing import Optional

import numpy as np

from pymrst.gridprocessing.grid import Cells, Faces, Grid, Nodes


def triangle_grid(p: np.ndarray, t: Optional[np.ndarray] = None) -> Grid:
    """Build a 2D grid from points p (m, 2) and triangles t (n, 3).

    t defaults to the Delaunay triangulation of p (scipy.spatial). Node
    indices in t are 0-based. Unused or duplicate nodes are dropped, as
    in MRST.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("triangle_grid requires an (m, 2) point array")
    if t is None:
        from scipy.spatial import Delaunay
        t = Delaunay(p).simplices
    t = np.asarray(t, dtype=np.int64)
    if t.ndim != 2 or t.shape[1] != 3:
        raise ValueError("triangle list must be (n, 3)")
    if t.size and (t.min() < 0 or t.max() >= p.shape[0]):
        raise ValueError("triangle list references invalid points")

    # Half-edges of every triangle in MRST's order: (1,2), (2,3), (3,1).
    half_edges = t[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2)
    ascending = half_edges[:, 0] <= half_edges[:, 1]
    sorted_edges = np.sort(half_edges, axis=1)

    unique_edges, cf = np.unique(sorted_edges, axis=0, return_inverse=True)
    cf = np.asarray(cf, dtype=np.int64).ravel()
    nf = unique_edges.shape[0]
    nc = t.shape[0]

    cell_no = np.repeat(np.arange(nc, dtype=np.int64), 3)
    neighbors = np.zeros((nf, 2), dtype=np.int64)
    np.add.at(neighbors, (cf, np.where(ascending, 0, 1)), cell_no + 1)
    neighbors -= 1

    coords = p
    face_nodes = unique_edges.reshape(-1)
    num_nodes = p.shape[0]

    used = np.zeros(num_nodes, dtype=bool)
    used[face_nodes] = True
    if used.sum() != num_nodes:
        coords, node_map = np.unique(coords, axis=0, return_inverse=True)
        face_nodes = np.asarray(node_map, dtype=np.int64).ravel()[face_nodes]
        num_nodes = coords.shape[0]

    return Grid(
        cells=Cells(
            num=nc,
            facePos=3 * np.arange(nc + 1, dtype=np.int64),
            faces=cf,
            indexMap=np.arange(nc, dtype=np.int64),
        ),
        faces=Faces(
            num=nf,
            nodePos=2 * np.arange(nf + 1, dtype=np.int64),
            nodes=face_nodes.astype(np.int64),
            neighbors=neighbors,
        ),
        nodes=Nodes(num=num_nodes, coords=np.array(coords, dtype=np.float64)),
        griddim=2,
        type=["triangleGrid"],
    )


if __name__ == "__main__":
    from pymrst.gridprocessing.geometry import compute_geometry

    def test_two_triangles():
        """Unit square split along the diagonal: 2 cells, 5 faces."""
        p = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        t = np.array([[0, 1, 2], [1, 3, 2]])
        g = triangle_grid(p, t)
        g.check()
        assert g.cells.num == 2 and g.faces.num == 5 and g.nodes.num == 4
        # the diagonal (1,2) is the only interior face
        interior = np.all(g.faces.neighbors >= 0, axis=1)
        assert interior.sum() == 1
        diag = np.flatnonzero(interior)[0]
        assert set(g.face_nodes(diag)) == {1, 2}
        assert set(g.faces.neighbors[diag]) == {0, 1}
        print("ok test_two_triangles")

    def test_geometry_covers_domain():
        """Areas of a triangulated rectangle sum to the rectangle's area."""
        x, y = np.meshgrid(np.linspace(0.0, 2.0, 5), np.linspace(0.0, 1.0, 4))
        p = np.column_stack([x.ravel(), y.ravel()])
        g = compute_geometry(triangle_grid(p))
        g.check()
        assert abs(g.cells.volumes.sum() - 2.0) < 1e-12
        assert np.all(g.cells.volumes > 0)
        print("ok test_geometry_covers_domain")

    def test_duplicate_nodes_are_merged():
        """MRST's uniquify step: duplicate coordinates merge (unused
        distinct points stay - MATLAB's unique keeps them)."""
        p = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
                      [0.0, 0.0], [5.0, 5.0]])
        t = np.array([[3, 1, 2]])  # references the duplicate of node 0
        g = triangle_grid(p, t)
        g.check()
        assert g.nodes.num == 4  # duplicate merged, unused (5,5) kept
        tri_coords = g.nodes.coords[np.unique(g.faces.nodes)]
        assert np.any(np.all(tri_coords == [0.0, 0.0], axis=1))
        print("ok test_duplicate_nodes_are_merged")

    def test_invalid_input_raises():
        """3D points and out-of-range triangle lists are rejected."""
        try:
            triangle_grid(np.zeros((4, 3)))
            raise SystemExit("should have raised on 3D points")
        except ValueError:
            pass
        try:
            triangle_grid(np.zeros((3, 2)), np.array([[0, 1, 3]]))
            raise SystemExit("should have raised on bad triangle list")
        except ValueError:
            pass
        print("ok test_invalid_input_raises")

    test_two_triangles()
    test_geometry_covers_domain()
    test_duplicate_nodes_are_merged()
    test_invalid_input_raises()
    print("\nAll tests passed!")
