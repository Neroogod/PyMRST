"""Grid geometry, ported from MRST `computeGeometry`."""

import warnings

import numpy as np

from pymrst.gridprocessing.grid_structure import Grid
from pymrst.utils.gridtools import mcolon


def compute_geometry(g: Grid) -> Grid:
    """Fill face and cell geometry fields on `g`."""
    if g.griddim == 1:
        _geom_1d(g)
    elif g.griddim == 2:
        assert g.spatial_dim in (2, 3), "2D grids live in 2 or 3 dimensions"
        _geom_2d(g)
    elif g.griddim == 3:
        _geom_3d(g)
    else:
        raise ValueError(f"unsupported grid dimension {g.griddim}")
    g.type = g.type + ["computeGeometry"]
    return g


def _accum(index: np.ndarray, values: np.ndarray, size: int) -> np.ndarray:
    """Sum rows of `values` into `size` bins."""
    if values.ndim == 1:
        return np.bincount(index, weights=values, minlength=size)
    return np.column_stack(
        [np.bincount(index, weights=values[:, col], minlength=size) for col in range(values.shape[1])]
    )


def _geom_3d(g: Grid) -> None:
    coords = g.nodes.coords
    node_pos = g.faces.nodePos
    face_nodes = g.faces.nodes
    nf, nc = g.faces.num, g.cells.num
    num_nodes = np.diff(node_pos)
    face_no = np.repeat(np.arange(nf, dtype=np.int64), num_nodes)
    p_centers = _accum(face_no, coords[face_nodes], nf) / num_nodes[:, None]

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

