"""Grid geometry, ported from MRST `computeGeometry`."""

import warnings

import numpy as np

from pymrst.gridprocessing.grid_structure import Grid
from pymrst.utils.gridtools import mcolon


def compute_geometry(g: Grid, *, cp_geometry: bool = False,
                     find_neighbors: bool = False, hinge_nodes=None) -> Grid:
    """Fill face and cell geometry fields on ``g``.

    With ``cp_geometry=True``, additionally derive MRST's nominal corner-point
    centers/extents when ``cells.cpnodes`` is available. For a 3D grid,
    ``hinge_nodes`` may be a mapping or object with ``faces`` and ``nodes``
    members. Its zero-based face indices and corresponding 3D coordinates
    replace the triangulation centers of non-planar faces.
    """
    hinges = _normalize_hinge_nodes(g, hinge_nodes)
    if find_neighbors or g.faces.neighbors is None:
        if g.griddim == 1:
            raise ValueError("neighbor reconstruction is supported in 2D/3D")
        _find_and_orient_neighbors(g)
    if g.griddim == 1:
        _geom_1d(g)
    elif g.griddim == 2:
        assert g.spatial_dim in (2, 3), "2D grids live in 2 or 3 dimensions"
        _geom_2d(g)
    elif g.griddim == 3:
        _geom_3d(g, hinges)
    else:
        raise ValueError(f"unsupported grid dimension {g.griddim}")
    if cp_geometry:
        if hasattr(g.cells, "cpnodes"):
            _compute_cp_geometry(g)
        else:
            warnings.warn(
                "corner-point geometry requested but cells.cpnodes is missing",
                RuntimeWarning,
                stacklevel=2,
            )
    g.type = g.type + ["computeGeometry"]
    return g


def _normalize_hinge_nodes(g: Grid, hinge_nodes):
    """Validate MRST ``hingenodes`` and return its two numeric arrays."""
    if hinge_nodes is None:
        return None
    if g.griddim != 3:
        raise ValueError("hinge_nodes is only supported for 3D grids")
    try:
        if isinstance(hinge_nodes, dict):
            faces = hinge_nodes["faces"]
            nodes = hinge_nodes["nodes"]
        else:
            faces = hinge_nodes.faces
            nodes = hinge_nodes.nodes
    except (KeyError, AttributeError) as exc:
        raise ValueError("hinge_nodes must define faces and nodes") from exc

    raw_faces = np.asarray(faces)
    if raw_faces.ndim > 2 or (raw_faces.ndim == 2 and 1 not in raw_faces.shape):
        raise ValueError("hinge_nodes.faces must be a vector")
    raw_faces = raw_faces.reshape(-1)
    if not np.issubdtype(raw_faces.dtype, np.integer):
        if not np.issubdtype(raw_faces.dtype, np.number) or not np.all(
            np.isfinite(raw_faces) & (raw_faces == np.floor(raw_faces))
        ):
            raise ValueError("hinge_nodes.faces must contain integer indices")
    faces = raw_faces.astype(np.int64)
    nodes = np.asarray(nodes, dtype=float)
    if faces.size == 0 and nodes.size == 0:
        nodes = nodes.reshape(0, 3)
    elif nodes.ndim == 1 and faces.size == 1:
        nodes = nodes.reshape(1, -1)
    if nodes.shape != (faces.size, 3):
        raise ValueError("hinge_nodes.nodes must have shape (number of faces, 3)")
    if np.any((faces < 0) | (faces >= g.faces.num)):
        raise ValueError("hinge_nodes.faces contains an out-of-range face index")
    if not np.all(np.isfinite(nodes)):
        raise ValueError("hinge_nodes.nodes must contain finite coordinates")
    return faces, nodes


def _find_and_orient_neighbors(g: Grid) -> None:
    """Port MRST ``findNeighbors`` plus ``findNormalDirections``."""
    cell_no = g.cell_no()
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64)
    order = np.argsort(cell_faces, kind="stable")
    sorted_faces = cell_faces[order]
    sorted_cells = cell_no[order]
    neighbors = np.full((g.faces.num, 2), -1, dtype=np.int64)
    for face in range(g.faces.num):
        cells = sorted_cells[sorted_faces == face]
        if cells.size > 2:
            raise ValueError(f"face {face} has more than two incident cells")
        neighbors[face, :cells.size] = cells
    g.faces.neighbors = neighbors

    counts = np.diff(g.faces.nodePos)
    face_no = np.repeat(np.arange(g.faces.num), counts)
    face_centers = _accum(
        face_no, g.nodes.coords[g.faces.nodes], g.faces.num
    ) / counts[:, None]
    per_cell = np.diff(g.cells.facePos)
    cell_centers = _accum(
        cell_no, face_centers[cell_faces], g.cells.num
    ) / per_cell[:, None]
    if g.griddim == 2:
        if g.spatial_dim != 2:
            raise ValueError("2D neighbor orientation requires 2D coordinates")
        edges = g.faces.nodes.reshape(-1, 2)
        length = g.nodes.coords[edges[:, 1]] - g.nodes.coords[edges[:, 0]]
        normal = np.column_stack([length[:, 1], -length[:, 0]])
        vector = face_centers[cell_faces] - cell_centers[cell_no]
        direction = np.sum(vector * normal[cell_faces], axis=1)
    else:
        first = g.faces.nodes[g.faces.nodePos[:-1]]
        second = g.faces.nodes[g.faces.nodePos[:-1] + 1]
        n1, n2 = g.nodes.coords[first], g.nodes.coords[second]
        v1 = face_centers[cell_faces] - cell_centers[cell_no]
        v2 = n1[cell_faces] - face_centers[cell_faces]
        v3 = n2[cell_faces] - n1[cell_faces]
        direction = np.sum(np.cross(v1, v2) * v3, axis=1)
    sign = np.where(neighbors[cell_faces, 0] == cell_no, 1.0, -1.0)
    score = _accum(cell_faces, direction * sign, g.faces.num)
    swap = score < 0
    g.faces.neighbors[swap] = g.faces.neighbors[swap, ::-1]


def _compute_cp_geometry(g: Grid) -> None:
    """Port of MRST private ``computeCpGeometry`` for preserved CP nodes."""
    from types import SimpleNamespace

    corners = np.asarray(g.nodes.coords)[np.asarray(g.cells.cpnodes)]
    nominal = np.array([
        [0, 2, 4, 6], [1, 3, 5, 7],
        [0, 1, 4, 5], [2, 3, 6, 7],
        [0, 1, 2, 3], [4, 5, 6, 7],
    ], dtype=np.int64)
    face_centers = corners[:, nominal].mean(axis=2)
    centroids = 0.5 * (face_centers[:, 0] + face_centers[:, 1])
    extent = np.column_stack([
        np.linalg.norm(face_centers[:, 2*d + 1] - face_centers[:, 2*d], axis=1)
        for d in range(g.griddim)
    ])
    cell_no = g.cell_no()
    tags = np.asarray(g.cells.tags, dtype=np.int64)
    half_face_centers = face_centers[cell_no, tags]
    g.cells.cpgeometry = SimpleNamespace(
        centroids=centroids,
        extent=extent,
        facecentroids=half_face_centers,
    )


def _accum(index: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    """Sum rows of `values` into `size` bins."""
    if values.ndim == 1:
        return np.bincount(index, weights=values, minlength=size)
    return np.column_stack(
        [np.bincount(index, weights=values[:, col], minlength=size) for col in range(values.shape[1])]
    )


def _geom_3d(g: Grid, hinge_nodes=None) -> None:
    coords = g.nodes.coords
    node_pos = g.faces.nodePos
    face_nodes = g.faces.nodes
    nf, nc = g.faces.num, g.cells.num
    num_nodes = np.diff(node_pos)
    face_no = np.repeat(np.arange(nf, dtype=np.int64), num_nodes)
    p_centers = _accum(face_no, coords[face_nodes], nf) / num_nodes[:, None]
    if hinge_nodes is not None:
        hinge_faces, hinge_coords = hinge_nodes
        for face, coordinate in zip(hinge_faces, hinge_coords):
            p_centers[face] = coordinate

    nxt = np.arange(1, face_nodes.size + 1, dtype=np.int64)
    nxt[node_pos[1:] - 1] = node_pos[:-1]
    a = coords[face_nodes]
    b = coords[face_nodes[nxt]]
    pc = p_centers[face_no]
    sub_normals = np.cross(b - a, pc - a) / 2.0
    sub_areas = np.sqrt(np.sum(sub_normals**2, axis=1))
    sub_centroids = (a + b + pc) / 3.0

    face_normals = _accum(face_no, sub_normals, nf)
    sub_signs = np.sign(np.sum(sub_normals * face_normals[face_no], axis=1))
    face_areas = _accum(face_no, sub_areas, nf)
    face_centroids = _face_centroids(face_areas, sub_areas, sub_centroids, face_no, p_centers, nf)

    cell_no = g.cell_no()
    cell_faces = g.cells.faces
    n_half_faces = np.diff(g.cells.facePos)
    c_centers = _accum(cell_no, face_centroids[cell_faces], nc) / n_half_faces[:, None]

    tri_ix = mcolon(node_pos[cell_faces], node_pos[cell_faces + 1])
    half_face_no = np.repeat(np.arange(cell_faces.size, dtype=np.int64), num_nodes[cell_faces])
    cell_no_tri = cell_no[half_face_no]
    rel_sub_centroids = sub_centroids[tri_ix] - c_centers[cell_no_tri]
    cell_face_sign = np.where(g.faces.neighbors[cell_faces, 0] == cell_no, 1.0, -1.0)
    out_normals = sub_normals[tri_ix] * (
        sub_signs[tri_ix] * cell_face_sign[half_face_no]
    )[:, None]

    tet_volumes = np.sum(rel_sub_centroids * out_normals, axis=1) / 3.0
    tet_centroids = 0.75 * rel_sub_centroids
    volumes = _accum(cell_no_tri, tet_volumes, nc)
    rel_centroids = _accum(cell_no_tri, tet_volumes[:, None] * tet_centroids, nc)
    rel_centroids /= volumes[:, None]

    g.faces.areas = face_areas
    g.faces.normals = face_normals
    g.faces.centroids = face_centroids
    g.cells.volumes = volumes
    g.cells.centroids = rel_centroids + c_centers


def _face_centroids(
    face_areas: np.ndarray,
    sub_areas: np.ndarray,
    sub_centroids: np.ndarray,
    face_no: np.ndarray,
    p_centers: np.ndarray,
    nf: int,
) -> np.ndarray:
    centroids = _accum(face_no, sub_areas[:, None] * sub_centroids, nf)
    degenerate = ~(face_areas > 0)
    if np.any(degenerate):
        warnings.warn(f"{int(degenerate.sum())} faces with non-positive area detected")
    centroids[~degenerate] /= face_areas[~degenerate, None]
    centroids[degenerate] = p_centers[degenerate]
    return centroids


def _geom_2d(g: Grid) -> None:
    coords = g.nodes.coords
    nf, nc = g.faces.num, g.cells.num
    dim = coords.shape[1]
    edges = g.faces.nodes.reshape(-1, 2)
    n1, n2 = coords[edges[:, 0]], coords[edges[:, 1]]
    lengths = n2 - n1

    face_areas = np.sqrt(np.sum(lengths**2, axis=1))
    face_centroids = (n1 + n2) / 2.0
    face_normals = _face_normals_2d(lengths, nf, dim)
    cell_no = g.cell_no()
    cell_faces = g.cells.faces
    n_half_faces = np.diff(g.cells.facePos)
    c_centers = _accum(cell_no, face_centroids[cell_faces], nc) / n_half_faces[:, None]

    cell_edges = edges[cell_faces].copy()
    flip = g.faces.neighbors[cell_faces, 1] == cell_no
    cell_edges[flip] = cell_edges[flip][:, ::-1]
    cell_centers = c_centers[cell_no]
    a = coords[cell_edges[:, 0]] - cell_centers
    b = coords[cell_edges[:, 1]] - cell_centers
    sub_areas = _sub_areas_2d(a, b, dim)
    sub_centroids = (cell_centers + 2.0 * face_centroids[cell_faces]) / 3.0

    volumes = _accum(cell_no, sub_areas, nc)
    centroids = _accum(cell_no, sub_areas[:, None] * sub_centroids, nc)
    centroids /= volumes[:, None]
    g.faces.areas = face_areas
    g.faces.normals = face_normals
    g.faces.centroids = face_centroids
    g.cells.volumes = volumes
    g.cells.centroids = centroids


def _face_normals_2d(lengths: np.ndarray, nf: int, dim: int) -> np.ndarray:
    if dim == 2:
        return np.column_stack([lengths[:, 1], -lengths[:, 0]])
    up = np.tile([0.0, 0.0, 1.0], (nf, 1))
    return np.cross(lengths, up)


def _sub_areas_2d(a: np.ndarray, b: np.ndarray, dim: int) -> np.ndarray:
    if dim == 2:
        return np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / 2.0
    return np.sqrt(np.sum(np.cross(a, b) ** 2, axis=1)) / 2.0


def _geom_1d(g: Grid) -> None:
    coords = g.nodes.coords.ravel()
    g.faces.areas = np.ones(g.faces.num)
    g.faces.normals = np.ones((g.faces.num, 1))
    g.faces.centroids = coords.reshape(-1, 1)
    g.cells.volumes = np.diff(coords)
    cell_no = g.cell_no()
    g.cells.centroids = (_accum(cell_no, coords[g.cells.faces], g.cells.num) / 2.0).reshape(-1, 1)


if __name__ == "__main__":
    from pymrst.gridprocessing.tensor_grid import cart_grid, tensor_grid

    def test_cart_grid_exact_geometry() -> None:
        grid = compute_geometry(cart_grid([3, 4, 5], [300.0, 200.0, 100.0]))
        dx, dy, dz = 100.0, 50.0, 20.0
        assert np.allclose(grid.cells.volumes, dx * dy * dz, rtol=1e-14)
        assert np.allclose(grid.cells.centroids[0], [dx / 2, dy / 2, dz / 2])
        x_faces = slice(0, 4 * 4 * 5)
        assert np.allclose(grid.faces.areas[x_faces], dy * dz, rtol=1e-14)
        assert np.allclose(grid.faces.normals[x_faces], [dy * dz, 0.0, 0.0])
        assert np.allclose(np.linalg.norm(grid.faces.normals, axis=1), grid.faces.areas, rtol=1e-13)
        print("ok test_cart_grid_exact_geometry")

    def test_tensor_grid_volumes() -> None:
        x = np.array([0.0, 1.0, 2.5, 5.0, 10.0])
        y = np.array([0.0, 2.0, 3.0, 7.0])
        z = np.array([0.0, 1.0, 4.0])
        grid = compute_geometry(tensor_grid(x, y, z))
        expected = (
            np.diff(x)[:, None, None]
            * np.diff(y)[None, :, None]
            * np.diff(z)[None, None, :]
        ).ravel(order="F")
        assert np.allclose(grid.cells.volumes, expected, rtol=1e-13)
        assert np.isclose(grid.cells.volumes.sum(), 10.0 * 7.0 * 4.0)
        print("ok test_tensor_grid_volumes")

    def test_perturbed_grid_is_consistent() -> None:
        rng = np.random.default_rng(42)
        grid = cart_grid([4, 4, 4], [1.0, 1.0, 1.0])
        points = grid.nodes.coords
        interior = np.all((points > 1e-12) & (points < 1.0 - 1e-12), axis=1)
        points[interior] += 0.05 * (rng.random((int(interior.sum()), 3)) - 0.5)
        compute_geometry(grid)
        assert np.isclose(grid.cells.volumes.sum(), 1.0, rtol=1e-12)
        assert np.all(grid.cells.volumes > 0)
        print("ok test_perturbed_grid_is_consistent")

    def test_normal_orientation() -> None:
        grid = compute_geometry(cart_grid([3, 3, 3]))
        internal = np.all(grid.faces.neighbors >= 0, axis=1)
        delta = (
            grid.cells.centroids[grid.faces.neighbors[internal, 1]]
            - grid.cells.centroids[grid.faces.neighbors[internal, 0]]
        )
        assert np.all(np.sum(delta * grid.faces.normals[internal], axis=1) > 0)
        print("ok test_normal_orientation")

    def test_2d_geometry() -> None:
        grid = compute_geometry(cart_grid([5, 7], [10.0, 21.0]))
        assert np.allclose(grid.cells.volumes, 2.0 * 3.0, rtol=1e-14)
        assert np.allclose(grid.cells.centroids[0], [1.0, 1.5])
        assert np.isclose(grid.cells.volumes.sum(), 210.0)
        print("ok test_2d_geometry")

    def test_1d_geometry() -> None:
        grid = compute_geometry(tensor_grid(np.array([0.0, 1.0, 2.5, 5.0])))
        assert np.allclose(grid.cells.volumes, [1.0, 1.5, 2.5])
        assert np.allclose(grid.cells.centroids.ravel(), [0.5, 1.75, 3.75])
        print("ok test_1d_geometry")

    test_cart_grid_exact_geometry()
    test_tensor_grid_volumes()
    test_perturbed_grid_is_consistent()
    test_normal_orientation()
    test_2d_geometry()
    test_1d_geometry()
    print("\nAll tests passed!")
