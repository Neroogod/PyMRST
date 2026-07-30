import numpy as np

from pymrst.gridprocessing import (
    cart_grid,
    compute_geometry,
    mark_cut_grids,
)


def _vertical_cut_status(grid):
    neighbors = grid.faces.neighbors
    status = np.zeros(grid.faces.num, dtype=np.int64)
    # In a 4x2x1 grid, separate global i columns 1 and 2 (zero-based).
    internal = np.all(neighbors >= 0, axis=1)
    left_i = neighbors[:, 0] % 4
    right_i = np.maximum(neighbors[:, 1], 0) % 4
    status[internal & (left_i == 1) & (right_i == 2)] = 3
    return status


def test_mark_cut_grids_labels_both_sides_of_plane():
    grid = cart_grid([4, 2, 1])
    labels = mark_cut_grids(grid, _vertical_cut_status(grid))
    assert labels.tolist() == [0, 0, 1, 1, 0, 0, 1, 1]


def test_legacy_mode_floods_only_start_side():
    grid = compute_geometry(cart_grid([4, 2, 1]))
    status = _vertical_cut_status(grid)
    assert mark_cut_grids(
        grid, status, legacy=True, start=0
    ).tolist() == [1, 1, 0, 0, 1, 1, 0, 0]
    assert mark_cut_grids(
        grid, status, legacy=True, start=grid.cells.centroids[3]
    ).tolist() == [0, 0, 1, 1, 0, 0, 1, 1]


def test_every_cell_is_separate_when_all_internal_faces_are_cut():
    grid = cart_grid([3, 1, 1])
    status = np.zeros(grid.faces.num, dtype=np.int64)
    status[np.all(grid.faces.neighbors >= 0, axis=1)] = 3
    labels = mark_cut_grids(grid, status)
    assert np.unique(labels).size == grid.cells.num
