import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry


@pytest.mark.parametrize("dims", ([3, 2], [3, 2, 2]))
def test_compute_geometry_reconstructs_exact_cartesian_neighbors(dims):
    grid = cart_grid(dims)
    expected = grid.faces.neighbors.copy()
    grid.faces.neighbors = np.full_like(expected, -1)
    compute_geometry(grid, find_neighbors=True)
    assert np.array_equal(grid.faces.neighbors, expected)


def test_compute_geometry_reconstructs_when_neighbors_missing():
    grid = cart_grid([2, 2, 1])
    expected = grid.faces.neighbors.copy()
    grid.faces.neighbors = None
    compute_geometry(grid)
    assert np.array_equal(grid.faces.neighbors, expected)


def test_compute_geometry_find_neighbors_rejects_nonmanifold_face():
    grid = cart_grid([3, 1])
    shared = int(np.flatnonzero(np.all(grid.faces.neighbors >= 0, axis=1))[0])
    grid.cells.faces[grid.cells.facePos[2]] = shared
    with pytest.raises(ValueError, match="more than two"):
        compute_geometry(grid, find_neighbors=True)
