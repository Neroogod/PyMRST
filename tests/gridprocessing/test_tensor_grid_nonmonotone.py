import numpy as np
import pytest

from pymrst.gridprocessing import tensor_grid


def test_tensor_grid_truncates_nonmonotone_1d_coordinates():
    with pytest.warns(RuntimeWarning, match="nonmonotone"):
        grid = tensor_grid(np.array([0.0, 2.0, 1.0, 3.0]))
    assert np.array_equal(grid.nodes.coords.ravel(), [0.0, 2.0, 3.0])
    assert grid.cells.num == 2


def test_tensor_grid_truncates_duplicates_repeatedly():
    with pytest.warns(RuntimeWarning):
        grid = tensor_grid(np.array([0.0, 1.0, 1.0, 0.5, 2.0]))
    assert np.array_equal(grid.nodes.coords.ravel(), [0.0, 1.0, 2.0])


def test_tensor_grid_applies_truncation_independently_per_axis():
    with pytest.warns(RuntimeWarning) as caught:
        grid = tensor_grid(
            np.array([0.0, 2.0, 1.0, 3.0]),
            np.array([0.0, 1.0, 1.0, 4.0]),
            np.array([0.0, 5.0, 3.0, 8.0]),
            cellnodes=True,
        )
    assert len(caught) == 3
    assert grid.cartDims == (2, 2, 2)
    assert grid.cells.num == 8
    assert grid.cellNodes.shape == (8, 8)


def test_tensor_grid_rejects_axis_that_cannot_form_a_cell():
    with pytest.warns(RuntimeWarning):
        with pytest.raises(ValueError, match="fewer than two"):
            tensor_grid(np.array([3.0, 2.0, 1.0]))
