"""Grid from points and a tetrahedron list, ported from MRST tetrahedralGrid.m.

Each tetrahedron becomes a cell with four triangular faces. Tetrahedra are
first re-oriented (nodes 1 and 2 swapped where the triple product is
positive), so that the parity of each face's node-sort permutation decides
the neighbor column, exactly as MATLAB's `any(diff(i)' == 1, 2)` does.
"""

from typing import Optional

import numpy as np

from pymrst.gridprocessing.grid import Cells, Faces, Grid, Nodes


def tetrahedral_grid(p: np.ndarray, t: Optional[np.ndarray] = None) -> Grid:
    """Build a 3D grid from points p (m, 3) and tetrahedra t (n, 4).

    t defaults to the Delaunay tessellation of p (scipy.spatial). Node
    indices in t are 0-based. Unused or duplicate nodes are dropped, as
    in MRST.
    """
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError("tetrahedral_grid requires an (m, 3) point array")
    if t is None:
        from scipy.spatial import Delaunay
        t = Delaunay(p).simplices
    t = np.asarray(t, dtype=np.int64)
    if t.ndim != 2 or t.shape[1] != 4:
        raise ValueError("tetrahedron list must be (n, 4)")
    if t.size and (t.min() < 0 or t.max() >= p.shape[0]):
        raise ValueError("tetrahedron list references invalid points")

    t = _permute_nodes(t, p)

    # Faces of every tetrahedron in MRST's order: (2,1,3), (1,2,4), (3,1,4), (2,3,4).
    tri = t[:, [1, 0, 2, 0, 1, 3, 2, 0, 3, 1, 2, 3]].reshape(-1, 3)
    perm = np.argsort(tri, axis=1)
    sorted_tri = np.take_along_axis(tri, perm, axis=1)
    # Even permutations (the cyclic ones) land the cell in neighbor column 2.
    even = ((perm[:, 0] + 1) % 3) == perm[:, 1]

    unique_tri, cf = np.unique(sorted_tri, axis=0, return_inverse=True)
    cf = np.asarray(cf, dtype=np.int64).ravel()
    nf = unique_tri.shape[0]
    nc = t.shape[0]

    cell_no = np.repeat(np.arange(nc, dtype=np.int64), 4)
    neighbors = np.zeros((nf, 2), dtype=np.int64)
    np.add.at(neighbors, (cf, np.where(even, 1, 0)), cell_no + 1)
    neighbors -= 1

    coords = p
    face_nodes = unique_tri.reshape(-1)
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
            facePos=4 * np.arange(nc + 1, dtype=np.int64),
            faces=cf,
            indexMap=np.arange(nc, dtype=np.int64),
        ),
        faces=Faces(
            num=nf,
            nodePos=3 * np.arange(nf + 1, dtype=np.int64),
            nodes=face_nodes.astype(np.int64),
            neighbors=neighbors,
        ),
        nodes=Nodes(num=num_nodes, coords=np.array(coords, dtype=np.float64)),
        griddim=3,
        type=["tetrahedralGrid"],
    )


def _permute_nodes(t: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Swap nodes 1 and 2 of tetrahedra whose triple product is positive."""
    d = p[t[:, 1:]] - p[t[:, [0]]]
    v = np.sum(np.cross(d[:, 0], d[:, 1]) * d[:, 2], axis=1)
    t = t.copy()
    t[v > 0, 0:2] = t[v > 0, 1::-1]
    return t


if __name__ == "__main__":
    from pymrst.gridprocessing.geometry import compute_geometry

    def _cube_points() -> np.ndarray:
        x, y, z = np.meshgrid([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])
        return np.column_stack([x.ravel(), y.ravel(), z.ravel()])

    def test_single_tet_both_orientations():
        """A tet and its mirror image both give one positive-volume cell."""
        p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        for t in (np.array([[0, 1, 2, 3]]), np.array([[1, 0, 2, 3]])):
            g = compute_geometry(tetrahedral_grid(p, t))
            g.check()
            assert g.cells.num == 1 and g.faces.num == 4
            assert abs(g.cells.volumes[0] - 1.0 / 6.0) < 1e-14
        print("ok test_single_tet_both_orientations")

    def test_cube_tessellation_fills_cube():
        """Delaunay tets of the unit cube's corners have total volume 1."""
        g = compute_geometry(tetrahedral_grid(_cube_points()))
        g.check()
        assert abs(g.cells.volumes.sum() - 1.0) < 1e-12
        assert np.all(g.cells.volumes > 0)
        print("ok test_cube_tessellation_fills_cube")

    def test_shared_face_neighbors():
        """Two tets sharing a face are neighbors across exactly one face."""
        p = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        t = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
        g = tetrahedral_grid(p, t)
        g.check()
        interior = np.all(g.faces.neighbors >= 0, axis=1)
        assert interior.sum() == 1 and g.faces.num == 7
        assert set(g.faces.neighbors[np.flatnonzero(interior)[0]]) == {0, 1}
        print("ok test_shared_face_neighbors")

    def test_invalid_input_raises():
        """2D points are rejected."""
        try:
            tetrahedral_grid(np.zeros((4, 2)))
            raise SystemExit("should have raised on 2D points")
        except ValueError:
            pass
        print("ok test_invalid_input_raises")

    test_single_tet_both_orientations()
    test_cube_tessellation_fills_cube()
    test_shared_face_neighbors()
    test_invalid_input_raises()
    print("\nAll tests passed!")
