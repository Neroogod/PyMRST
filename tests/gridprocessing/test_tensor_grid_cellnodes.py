import numpy as np

from pymrst.gridprocessing import cart_grid, tensor_grid


def test_tensor_grid_cellnodes_3d_local_corner_order():
    grid = cart_grid([2, 2, 1], cellnodes=True)
    assert grid.cellNodes.shape == (4, 8)
    assert np.array_equal(grid.cellNodes[0], [0, 1, 3, 4, 9, 10, 12, 13])
    assert grid.cellNodes.min() >= 0
    assert grid.cellNodes.max() < grid.nodes.num


def test_tensor_grid_cellnodes_2d_local_corner_order():
    grid = cart_grid([2, 2], cellnodes=True)
    assert np.array_equal(grid.cellNodes, [
        [0, 1, 3, 4], [1, 2, 4, 5],
        [3, 4, 6, 7], [4, 5, 7, 8],
    ])


def test_tensor_grid_cellnodes_1d_matches_mrst_representatives():
    grid = tensor_grid(np.array([0.0, 1.0, 3.0, 6.0]), cellnodes=True)
    assert np.array_equal(grid.cellNodes, [[0], [1], [2]])


def test_cellnodes_absent_by_default():
    assert not hasattr(cart_grid([1, 1, 1]), "cellNodes")
