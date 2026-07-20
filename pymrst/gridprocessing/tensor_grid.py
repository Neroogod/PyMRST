"""Cartesian grid constructors, ported from MRST tensorGrid.m and cartGrid.m.

Node, face, and cell numbering reproduces MATLAB MRST exactly (0-based):
the i-index runs fastest, then j, then k, and faces are numbered x-faces
first, then y-faces, then z-faces. All MATLAB column-major reshapes are
mirrored with order='F'.
"""

from typing import Optional, Sequence

import numpy as np

from pymrst.gridprocessing.grid import Cells, Faces, Grid, Nodes


def tensor_grid(
    x: np.ndarray,
    y: Optional[np.ndarray] = None,
    z: Optional[np.ndarray] = None,
    depthz: Optional[np.ndarray] = None,
) -> Grid:
    """Construct a Cartesian grid with variable cell sizes from vertex vectors.

    Cell (i, j, k) spans [x[i], x[i+1]] x [y[j], y[j+1]] x [z[k], z[k+1]].
    depthz (3D only) shifts nodal depths of every pillar; shape (nx+1, ny+1).
    """
    if z is not None:
        g = _tensor_grid_3d(_monotone(x), _monotone(y), _monotone(z), depthz)
    elif y is not None:
        assert depthz is None, "depthz is only meaningful for 3D grids"
        g = _tensor_grid_2d(_monotone(x), _monotone(y))
    else:
        assert depthz is None, "depthz is only meaningful for 3D grids"
        g = _tensor_grid_1d(_monotone(x))
    g.type = ["tensorGrid"]
    g.griddim = len(g.cartDims)
    return g


def cart_grid(
    celldim: Sequence[int],
    physdims: Optional[Sequence[float]] = None,
    depthz: Optional[np.ndarray] = None,
) -> Grid:
    """Construct a uniform Cartesian grid: celldim cells over physdims meters.

    physdims defaults to celldim (unit cells), matching MRST's cartGrid.
    """
    celldim = tuple(int(d) for d in celldim)
    assert all(d > 0 for d in celldim), "celldim must be positive"
    if physdims is None:
        physdims = celldim
    coords = [
        np.linspace(0.0, float(p), d + 1) for p, d in zip(physdims, celldim)
    ]
    g = tensor_grid(*coords, depthz=depthz)
    g.type.append("cartGrid")
    return g


def _monotone(v: np.ndarray) -> np.ndarray:
    """Return v as a float vector, requiring strictly increasing entries."""
    v = np.asarray(v, dtype=np.float64).ravel()
    if not np.all(np.diff(v) > 0):
        raise ValueError("coordinate vector must be strictly increasing")
    return v


def _tensor_grid_3d(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, depthz: Optional[np.ndarray]
) -> Grid:
    sx, sy, sz = x.size - 1, y.size - 1, z.size - 1

    if depthz is None:
        depthz = np.zeros((sx + 1, sy + 1))
    depthz = np.asarray(depthz, dtype=np.float64)
    assert depthz.shape == (sx + 1, sy + 1), "depthz must be (nx+1, ny+1)"

    num_c = sx * sy * sz
    num_n = (sx + 1) * (sy + 1) * (sz + 1)
    num_fx = (sx + 1) * sy * sz
    num_fy = sx * (sy + 1) * sz
    num_fz = sx * sy * (sz + 1)
    num_f = num_fx + num_fy + num_fz

    xc, yc, zc = np.meshgrid(x, y, z, indexing="ij")
    zc = zc + depthz[:, :, None]
    coords = np.column_stack(
        [xc.ravel(order="F"), yc.ravel(order="F"), zc.ravel(order="F")]
    )

    # Face -> node connectivity; node order fixes the normal orientation.
    n = np.arange(num_n, dtype=np.int64).reshape((sx + 1, sy + 1, sz + 1), order="F")

    def flat(a: np.ndarray) -> np.ndarray:
        return a.ravel(order="F")

    face_nodes_x = np.column_stack([
        flat(n[:, :sy, :sz]), flat(n[:, 1:, :sz]),
        flat(n[:, 1:, 1:]), flat(n[:, :sy, 1:]),
    ]).ravel()
    face_nodes_y = np.column_stack([
        flat(n[:sx, :, :sz]), flat(n[:sx, :, 1:]),
        flat(n[1:, :, 1:]), flat(n[1:, :, :sz]),
    ]).ravel()
    face_nodes_z = np.column_stack([
        flat(n[:sx, :sy, :]), flat(n[1:, :sy, :]),
        flat(n[1:, 1:, :]), flat(n[:sx, 1:, :]),
    ]).ravel()
    face_nodes = np.concatenate([face_nodes_x, face_nodes_y, face_nodes_z])

    # Cell -> face connectivity, tags 0..5 = W, E, S, N, T, B.
    fx = np.arange(num_fx, dtype=np.int64).reshape((sx + 1, sy, sz), order="F")
    fy = num_fx + np.arange(num_fy, dtype=np.int64).reshape((sx, sy + 1, sz), order="F")
    fz = num_fx + num_fy + np.arange(num_fz, dtype=np.int64).reshape(
        (sx, sy, sz + 1), order="F"
    )
    cell_faces = np.column_stack([
        flat(fx[:sx]), flat(fx[1:]),
        flat(fy[:, :sy]), flat(fy[:, 1:]),
        flat(fz[:, :, :sz]), flat(fz[:, :, 1:]),
    ]).ravel()
    tags = np.tile(np.arange(6, dtype=np.int64), num_c)

    # Face neighbors from a -1-padded cell index array.
    c = -np.ones((sx + 2, sy + 2, sz + 2), dtype=np.int64)
    c[1:-1, 1:-1, 1:-1] = np.arange(num_c, dtype=np.int64).reshape(
        (sx, sy, sz), order="F"
    )
    neighbors = np.vstack([
        np.column_stack([flat(c[:sx + 1, 1:sy + 1, 1:sz + 1]),
                         flat(c[1:, 1:sy + 1, 1:sz + 1])]),
        np.column_stack([flat(c[1:sx + 1, :sy + 1, 1:sz + 1]),
                         flat(c[1:sx + 1, 1:, 1:sz + 1])]),
        np.column_stack([flat(c[1:sx + 1, 1:sy + 1, :sz + 1]),
                         flat(c[1:sx + 1, 1:sy + 1, 1:])]),
    ])

    return Grid(
        cells=Cells(
            num=num_c,
            facePos=6 * np.arange(num_c + 1, dtype=np.int64),
            faces=cell_faces,
            indexMap=np.arange(num_c, dtype=np.int64),
            tags=tags,
        ),
        faces=Faces(
            num=num_f,
            nodePos=4 * np.arange(num_f + 1, dtype=np.int64),
            nodes=face_nodes,
            neighbors=neighbors,
            tag=np.zeros(num_f, dtype=np.int64),
        ),
        nodes=Nodes(num=num_n, coords=coords),
        griddim=3,
        cartDims=(sx, sy, sz),
    )


def _tensor_grid_2d(x: np.ndarray, y: np.ndarray) -> Grid:
    sx, sy = x.size - 1, y.size - 1

    num_c = sx * sy
    num_n = (sx + 1) * (sy + 1)
    num_fx = (sx + 1) * sy
    num_fy = sx * (sy + 1)
    num_f = num_fx + num_fy

    xc, yc = np.meshgrid(x, y, indexing="ij")
    coords = np.column_stack([xc.ravel(order="F"), yc.ravel(order="F")])

    n = np.arange(num_n, dtype=np.int64).reshape((sx + 1, sy + 1), order="F")

    def flat(a: np.ndarray) -> np.ndarray:
        return a.ravel(order="F")

    face_nodes_x = np.column_stack([flat(n[:, :sy]), flat(n[:, 1:])]).ravel()
    # y-face nodes reversed so normals point in positive j-direction
    face_nodes_y = np.column_stack([flat(n[1:, :]), flat(n[:sx, :])]).ravel()
    face_nodes = np.concatenate([face_nodes_x, face_nodes_y])

    fx = np.arange(num_fx, dtype=np.int64).reshape((sx + 1, sy), order="F")
    fy = num_fx + np.arange(num_fy, dtype=np.int64).reshape((sx, sy + 1), order="F")

    # Per-cell face order W, S, E, N with tags 0, 2, 1, 3 (as in MRST).
    cell_faces = np.column_stack([
        flat(fx[:sx]), flat(fy[:, :sy]), flat(fx[1:]), flat(fy[:, 1:]),
    ]).ravel()
    tags = np.tile(np.array([0, 2, 1, 3], dtype=np.int64), num_c)

    c = -np.ones((sx + 2, sy + 2), dtype=np.int64)
    c[1:-1, 1:-1] = np.arange(num_c, dtype=np.int64).reshape((sx, sy), order="F")
    neighbors = np.vstack([
        np.column_stack([flat(c[:sx + 1, 1:sy + 1]), flat(c[1:, 1:sy + 1])]),
        np.column_stack([flat(c[1:sx + 1, :sy + 1]), flat(c[1:sx + 1, 1:])]),
    ])

    return Grid(
        cells=Cells(
            num=num_c,
            facePos=4 * np.arange(num_c + 1, dtype=np.int64),
            faces=cell_faces,
            indexMap=np.arange(num_c, dtype=np.int64),
            tags=tags,
        ),
        faces=Faces(
            num=num_f,
            nodePos=2 * np.arange(num_f + 1, dtype=np.int64),
            nodes=face_nodes,
            neighbors=neighbors,
            tag=np.zeros(num_f, dtype=np.int64),
        ),
        nodes=Nodes(num=num_n, coords=coords),
        griddim=2,
        cartDims=(sx, sy),
    )


def _tensor_grid_1d(x: np.ndarray) -> Grid:
    sx = x.size - 1
    num_c, num_f = sx, sx + 1

    neighbors = np.column_stack([
        np.arange(num_f, dtype=np.int64) - 1,
        np.arange(num_f, dtype=np.int64),
    ])
    neighbors[-1, 1] = -1

    # Cell c is bounded by faces c (left, tag W) and c+1 (right, tag E).
    cell_faces = np.column_stack([
        np.arange(num_c, dtype=np.int64), np.arange(1, num_c + 1, dtype=np.int64)
    ]).ravel()
    tags = np.tile(np.array([0, 1], dtype=np.int64), num_c)

    return Grid(
        cells=Cells(
            num=num_c,
            facePos=2 * np.arange(num_c + 1, dtype=np.int64),
            faces=cell_faces,
            indexMap=np.arange(num_c, dtype=np.int64),
            tags=tags,
        ),
        faces=Faces(
            num=num_f,
            nodePos=np.arange(num_f + 1, dtype=np.int64),
            nodes=np.arange(num_f, dtype=np.int64),
            neighbors=neighbors,
            tag=np.zeros(num_f, dtype=np.int64),
        ),
        nodes=Nodes(num=num_f, coords=x.reshape(-1, 1)),
        griddim=1,
        cartDims=(sx,),
    )


if __name__ == "__main__":
    from pymrst.gridprocessing.grid import EAST, WEST

    def test_cart_grid_3d_counts():
        """3x4x5 grid has the MRST cell/face/node counts and valid topology."""
        g = cart_grid([3, 4, 5], [300.0, 200.0, 100.0])
        assert g.cells.num == 60
        assert g.nodes.num == 4 * 5 * 6
        assert g.faces.num == 4 * 4 * 5 + 3 * 5 * 5 + 3 * 4 * 6  # 227
        assert g.cartDims == (3, 4, 5)
        assert g.type == ["tensorGrid", "cartGrid"]
        g.check()
        print("ok test_cart_grid_3d_counts")

    def test_boundary_face_count_3d():
        """Boundary faces (one -1 neighbor) count matches the box surface."""
        g = cart_grid([3, 4, 5])
        n_boundary = int(np.sum(np.any(g.faces.neighbors == -1, axis=1)))
        assert n_boundary == 2 * (4 * 5 + 3 * 5 + 3 * 4)  # 94
        internal = g.faces.neighbors[np.all(g.faces.neighbors >= 0, axis=1)]
        assert internal.shape[0] == g.faces.num - n_boundary
        print("ok test_boundary_face_count_3d")

    def test_numbering_matches_mrst():
        """Node coords are i-fastest; cell 0's W/E faces sit at x=0 and x=100."""
        g = cart_grid([2, 2, 2], [200.0, 100.0, 50.0])
        assert np.array_equal(g.nodes.coords[0], [0.0, 0.0, 0.0])
        assert np.array_equal(g.nodes.coords[1], [100.0, 0.0, 0.0])  # i runs first
        assert np.array_equal(g.nodes.coords[3], [0.0, 50.0, 0.0])   # then j
        f = g.cell_faces(0)
        t = g.cells.tags[g.cells.facePos[0]:g.cells.facePos[1]]
        w_nodes = g.face_nodes(f[t == WEST][0])
        e_nodes = g.face_nodes(f[t == EAST][0])
        assert np.all(g.nodes.coords[w_nodes, 0] == 0.0)
        assert np.all(g.nodes.coords[e_nodes, 0] == 100.0)
        print("ok test_numbering_matches_mrst")

    def test_each_internal_face_shared_twice():
        """Internal faces appear in exactly two cells, boundary faces in one."""
        g = cart_grid([3, 3, 3])
        counts = np.bincount(g.cells.faces, minlength=g.faces.num)
        n_bdry = np.sum(np.any(g.faces.neighbors == -1, axis=1))
        assert np.all(np.isin(counts, [1, 2]))
        assert np.sum(counts == 1) == n_bdry
        print("ok test_each_internal_face_shared_twice")

    def test_depthz_shifts_pillars():
        """depthz adds pillar-wise depth to every z-coordinate."""
        depth = np.array([[10.0, 20.0], [30.0, 40.0]])
        g = tensor_grid(
            np.array([0.0, 1.0]), np.array([0.0, 1.0]),
            np.array([0.0, 1.0]), depthz=depth,
        )
        assert g.nodes.coords[0, 2] == 10.0   # pillar (0,0), top
        assert g.nodes.coords[3, 2] == 40.0   # pillar (1,1), top
        assert g.nodes.coords[4, 2] == 11.0   # pillar (0,0), bottom
        print("ok test_depthz_shifts_pillars")

    def test_2d_and_1d_grids():
        """2D and 1D variants have consistent topology and counts."""
        g2 = cart_grid([5, 7], [10.0, 20.0])
        assert g2.cells.num == 35 and g2.griddim == 2
        assert g2.faces.num == 6 * 7 + 5 * 8
        g2.check()
        g1 = tensor_grid(np.array([0.0, 1.0, 2.5, 5.0]))
        assert g1.cells.num == 3 and g1.faces.num == 4 and g1.griddim == 1
        assert np.array_equal(g1.faces.neighbors[0], [-1, 0])
        assert np.array_equal(g1.faces.neighbors[-1], [2, -1])
        g1.check()
        print("ok test_2d_and_1d_grids")

    def test_nonmonotone_rejected():
        """Non-increasing coordinate vectors raise ValueError."""
        try:
            tensor_grid(np.array([0.0, 2.0, 1.0]))
            raise SystemExit("expected ValueError")
        except ValueError:
            pass
        print("ok test_nonmonotone_rejected")

    test_cart_grid_3d_counts()
    test_boundary_face_count_3d()
    test_numbering_matches_mrst()
    test_each_internal_face_shared_twice()
    test_depthz_shifts_pillars()
    test_2d_and_1d_grids()
    test_nonmonotone_rejected()
    print("\nAll tests passed!")
