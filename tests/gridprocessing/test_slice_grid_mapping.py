import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.slice_grid import slice_grid


def test_slice_grid_returns_mrst_style_mapping_and_preserves_index_map():
    grid = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
    grid.cells.indexMap = np.array([8, 12])
    sliced, gix = slice_grid(
        grid, np.array([0.5, 0.0, 0.0]), normal=np.array([1.0, 0.0, 0.0]),
        return_mapping=True,
    )
    assert sliced.cells.num == 3
    assert np.array_equal(sliced.cells.indexMap, [8, 12, 8])
    assert np.array_equal(gix.parent.cells, [0, 1, 0])
    assert np.array_equal(gix.new.cells, [2, 1, 2])
    assert np.count_nonzero(gix.new.faces == 3) == 1
    assert gix.parent.faces[gix.new.faces == 3].item() == -1
    assert gix.old.cells.tolist() == [False, True]


def test_slice_grid_multiple_cuts_compose_parent_mapping():
    grid = cart_grid([2, 1, 1], [2.0, 1.0, 1.0])
    sliced, gix = slice_grid(
        grid,
        [np.array([0.4, 0.0, 0.0]), np.array([1.6, 0.0, 0.0])],
        normal=[np.array([1.0, 0.0, 0.0])] * 2,
        return_mapping=True,
    )
    sliced.check()
    assert sliced.cells.num == 4
    assert np.array_equal(gix.parent.cells, [0, 1, 0, 1])
    assert np.count_nonzero(gix.new.faces == 3) == 2
