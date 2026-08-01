import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.grid import extract_subgrid
from pymrst.gridprocessing.make_layered_grid import make_layered_grid
from pymrst.params import make_rock, vertical_well


def _rock(grid):
    return make_rock(grid, np.tile([1e-13, 1e-13, 1e-14], (grid.cells.num, 1)), .2)


def test_vertical_well_logical_and_top_cell_modes_agree():
    grid = compute_geometry(cart_grid([4, 3, 3], [40, 30, 30]))
    rock = _rock(grid)
    logical = vertical_well(None, grid, rock, 2, 1, [0, 2])[0]
    top = vertical_well(None, grid, rock, 6, k=[0, 2], top_cell=True)[0]
    np.testing.assert_array_equal(logical.cells, top.cells)


def test_vertical_well_empty_layers_mean_all_and_direction_is_forced():
    grid = compute_geometry(cart_grid([3, 2, 3], [30, 20, 30]))
    well = vertical_well(
        None, grid, _rock(grid), 1, 1, [], well_dir="x"
    )[0]
    np.testing.assert_array_equal(well.cells, [4, 10, 16])
    assert well.dir == "zzz"


def test_vertical_well_skips_inactive_completions_and_empty_is_noop():
    parent = compute_geometry(cart_grid([3, 2, 3], [30, 20, 30]))
    column = [4, 10, 16]
    keep = np.setdiff1d(np.arange(parent.cells.num), [10])
    grid = compute_geometry(extract_subgrid(parent, keep))
    well = vertical_well(None, grid, _rock(grid), 1, 1)[0]
    np.testing.assert_array_equal(grid.cells.indexMap[well.cells], [4, 16])

    empty_grid = compute_geometry(
        extract_subgrid(parent, np.setdiff1d(np.arange(parent.cells.num), column))
    )
    with pytest.warns(RuntimeWarning, match="Empty cell selection"):
        result = vertical_well(None, empty_grid, _rock(empty_grid), 1, 1)
    assert result == []


def test_vertical_well_top_cell_mode_supports_layered_grid():
    base = cart_grid([2, 2], [20, 20])
    grid = compute_geometry(make_layered_grid(base, [2.0, 3.0, 5.0]))
    well = vertical_well(
        None, grid, _rock(grid), 3, k=None, top_cell=True
    )[0]
    np.testing.assert_array_equal(well.cells, [3, 7, 11])


def test_vertical_well_validates_modes_and_layers():
    grid = compute_geometry(cart_grid([2, 2, 2]))
    rock = _rock(grid)
    with pytest.raises(ValueError, match="j is required"):
        vertical_well(None, grid, rock, 0)
    with pytest.raises(ValueError, match="must be omitted"):
        vertical_well(None, grid, rock, 0, 1, top_cell=True)
    with pytest.raises(IndexError, match="layer index"):
        vertical_well(None, grid, rock, 0, 0, [2])
