import warnings

import numpy as np
import pytest

from pymrst.gridprocessing import compute_geometry, process_grdecl
from pymrst.gridprocessing.deprecated import compute_geometry_orig
from pymrst.gridprocessing.testgrids import (
    cartesian_grdecl,
    create_pinched_column,
    cut_cell_grids,
    extruded_triangle_grid,
    make_model3,
    mark_cut_grids_example1,
    mark_cut_grids_example2,
    one_sloping_fault,
    pinch_middle_cell,
    pinched_layers_grdecl,
    pinched_node,
    raised_column,
    simple_grdecl,
    three_layers,
    twister,
)


@pytest.mark.parametrize("factory", [
    lambda: cartesian_grdecl([0, 1, 3], [0, 2], [0, 1]),
    lambda: simple_grdecl([4, 2, 2], .1),
    lambda: one_sloping_fault([4, 2, 2], 2),
    create_pinched_column,
    pinched_node,
    raised_column,
    lambda: three_layers(2, 2, 1),
    lambda: make_model3([4, 3, 2]),
    lambda: pinched_layers_grdecl([4, 3, 2], .02, seed=0),
])
def test_corner_point_fixture_processes_end_to_end(factory):
    grid = compute_geometry(process_grdecl(factory()))
    grid.check()
    assert grid.cells.num > 0
    assert np.all(np.isfinite(grid.cells.volumes))
    assert np.all(grid.cells.volumes > 0)


def test_pinch_middle_cell_all_combinations_process():
    fixtures = pinch_middle_cell(2)
    assert len(fixtures) == 6
    for deck in fixtures:
        process_grdecl(deck).check()


def test_twister_preserves_bounds_and_changes_interior():
    points = np.array([
        [0., 0.], [1., 0.], [0., 1.], [1., 1.], [.4, .6],
    ])
    out = twister(points)
    assert np.allclose(out[:4], points[:4])
    assert not np.allclose(out[-1], points[-1])
    assert np.allclose(out.min(axis=0), points.min(axis=0))
    assert np.allclose(out.max(axis=0), points.max(axis=0))


def test_deprecated_compute_geometry_orig_is_compatible():
    grid = process_grdecl(cartesian_grdecl([0, 1], [0, 1], [0, 1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        result = compute_geometry_orig(grid)
    assert np.allclose(result.cells.volumes, [1.0])


def test_cut_cell_grids_stress_case_conserves_volume():
    result = cut_cell_grids(2, seed=4)
    assert np.isclose(result["final_volume"], result["initial_volume"])
    result["grid"].check()


def test_extruded_triangle_grid_builds_valid_faulted_fixture():
    grid = extruded_triangle_grid(1000.0, seed=2)
    grid.check()
    assert grid.griddim == 3
    assert grid.cells.num > 0


def test_mark_cut_grids_examples_return_region_labels():
    for grid, labels in (
        mark_cut_grids_example1(2),
        mark_cut_grids_example2(2, 2, seed=3),
    ):
        grid.check()
        assert labels.shape == (grid.cells.num,)
        assert np.unique(labels).size >= 2
