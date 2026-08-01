from __future__ import annotations

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.utils.equil.test_simple_equilibrium import test_simple_equilibrium_case as _equilibrium_case
from pymrst.utils.gridtools.examples.grid_helpers_example import grid_helpers_example
from pymrst.utils.gridtools.grid_add_helpers import grid_add_helpers
from pymrst.utils.gridtools.tests.plot_single_cell_info import plot_single_cell_info
from pymrst.utils.gridtools.tests.plot_single_face_info import plot_single_face_info
from pymrst.utils.gridtools.tests.test_gridtools import test_gridtools_case as _gridtools_case


def test_equilibrium_example_covers_vertical_and_directional_contacts():
    result = _equilibrium_case()
    assert result["vertical"].shape == (100, 3)


def test_grid_helpers_example_exposes_mrst_query_closures():
    result = grid_helpers_example(shape=(3, 2, 1), plot=False)
    grid = result["grid"]
    assert "gridAddHelpers" in grid.type
    assert result["nodes"].size and result["faces"].size
    assert np.array_equal(grid.helpers.cellNo(), grid.cell_no())


def test_grid_add_helpers_requires_computed_geometry():
    grid = cart_grid([2, 1, 1])
    try:
        grid_add_helpers(grid)
    except ValueError as error:
        assert "compute_geometry" in str(error)
    else:
        raise AssertionError("grid_add_helpers accepted a grid without geometry")


def test_single_face_and_cell_diagnostics_match_topology():
    grid = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
    face = int(grid.cells.faces[0])
    face_info = plot_single_face_info(grid, face, from_cell=0)
    cell_info = plot_single_cell_info(grid, 0, plot_face_info=[0])
    assert face_info["nodes"].size == 4
    assert face_info["zero_length_segments"] == 0
    assert np.isclose(np.linalg.norm(face_info["unit_normal"]), 1.0)
    assert len(cell_info["details"]) == np.diff(grid.cells.facePos)[0]
    assert cell_info["plot_info"].sum() == 1


def test_mrst_gridtools_script_checks_pass_headlessly():
    result = _gridtools_case()
    assert result["nodes"].size > 0
    assert result["face_nodes"].size > 0
