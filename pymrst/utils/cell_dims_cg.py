"""Bounding-box dimensions of coarse cells (MRST ``utils/cellDimsCG.m``).

The extent of each named cell, taken as the spread of its *face centroids*
along each axis. `cellDims` measures a fine Cartesian cell from its corner
nodes; a coarse cell has no such nodes, so this uses what it does have.

Face centroids under-measure. The spread of centroids across a unit cube is
1, but across a shape whose faces are not axis-aligned it is less than the
true extent, because a centroid sits inside the face rather than at its
corner. That is MRST's definition and the well-index code is calibrated to it.

In 2-D, `dz` is 1 rather than 0 - the convention that makes an areal grid
behave as a unit-thickness slab in volume calculations.
"""

import numpy as np


def cell_dims_cg(g, cells=None):
    """Return ``(dx, dy, dz)`` for each cell in ``cells``."""
    cells = (np.arange(g.cells.num) if cells is None
             else np.atleast_1d(np.asarray(cells, dtype=np.int64)).ravel())
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64)
    cell_faces = cell_faces[:, 0] if cell_faces.ndim > 1 else cell_faces
    centroids = np.asarray(g.faces.centroids, dtype=float)
    dim = centroids.shape[1]

    out = np.ones((cells.size, 3))
    for i, cell in enumerate(cells):
        faces = cell_faces[face_pos[cell]:face_pos[cell + 1]]
        block = centroids[faces]
        delta = block.max(axis=0) - block.min(axis=0)
        out[i, :dim] = delta
        if dim <= 2:
            out[i, 2] = 1.0
    return out[:, 0], out[:, 1], out[:, 2]


cellDimsCG = cell_dims_cg


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_a_unit_cell_measures_one_by_one() -> None:
        g = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
        dx, dy, dz = cell_dims_cg(g)
        assert np.allclose(dx, 1.0) and np.allclose(dy, 1.0), (dx, dy)
        print("ok test_a_unit_cell_measures_one_by_one")

    def test_two_dimensional_thickness_is_one_not_zero() -> None:
        g = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        assert np.allclose(cell_dims_cg(g)[2], 1.0)
        print("ok test_two_dimensional_thickness_is_one_not_zero")

    def test_a_stretched_grid_reports_its_stretch() -> None:
        g = compute_geometry(cart_grid([2, 2], [4.0, 10.0]))
        dx, dy, _ = cell_dims_cg(g)
        assert np.allclose(dx, 2.0) and np.allclose(dy, 5.0), (dx, dy)
        print("ok test_a_stretched_grid_reports_its_stretch")

    def test_three_dimensions_measures_all_three() -> None:
        g = compute_geometry(cart_grid([1, 1, 1], [2.0, 3.0, 4.0]))
        dx, dy, dz = cell_dims_cg(g)
        assert np.allclose([dx[0], dy[0], dz[0]], [2.0, 3.0, 4.0])
        print("ok test_three_dimensions_measures_all_three")

    def test_a_subset_returns_only_those_cells() -> None:
        g = compute_geometry(cart_grid([4, 4], [4.0, 4.0]))
        dx, _, _ = cell_dims_cg(g, [0, 5])
        assert dx.size == 2, dx
        print("ok test_a_subset_returns_only_those_cells")

    def test_a_single_cell_is_accepted() -> None:
        g = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        assert cell_dims_cg(g, 1)[0].size == 1
        print("ok test_a_single_cell_is_accepted")

    test_a_unit_cell_measures_one_by_one()
    test_two_dimensional_thickness_is_one_not_zero()
    test_a_stretched_grid_reports_its_stretch()
    test_three_dimensions_measures_all_three()
    test_a_subset_returns_only_those_cells()
    test_a_single_cell_is_accepted()
    print("\nAll tests passed!")
