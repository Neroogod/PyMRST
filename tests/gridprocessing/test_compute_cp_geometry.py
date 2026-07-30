import numpy as np
import pytest

from pymrst.gridprocessing import compute_geometry, process_grdecl
from pymrst.gridprocessing.testgrids import simple_grdecl


def test_compute_cp_geometry_regular_grid_centers_extents_and_half_faces():
    grid = process_grdecl(
        simple_grdecl([2, 1, 1], 0.0, flat=True),
        preserve_cp_nodes=True,
        split_disconnected=False,
    )
    grid = compute_geometry(grid, cp_geometry=True)
    cp = grid.cells.cpgeometry
    assert cp.centroids.shape == (2, 3)
    assert cp.extent.shape == (2, 3)
    assert cp.facecentroids.shape == (12, 3)
    corner_mean = grid.nodes.coords[grid.cells.cpnodes].mean(axis=1)
    assert np.allclose(cp.centroids, corner_mean)
    assert np.all(cp.extent > 0)
    assert np.isclose(cp.extent[:, 0].sum(), 1.0)
    assert np.allclose(cp.extent[0, 1:], cp.extent[1, 1:])


def test_compute_cp_geometry_uses_original_faulted_cell_corners():
    deck = simple_grdecl([2, 1, 2], 0.3, flat=True)
    grid = process_grdecl(
        deck, preserve_cp_nodes=True, split_disconnected=False
    )
    grid = compute_geometry(grid, cp_geometry=True)
    cp = grid.cells.cpgeometry
    assert np.all(np.isfinite(cp.centroids))
    assert np.all(cp.extent > 0)
    assert cp.facecentroids.shape[0] == grid.cells.faces.size


def test_compute_cp_geometry_warns_without_preserved_corner_nodes():
    grid = process_grdecl(
        simple_grdecl([1, 1, 1], 0.0, flat=True),
        split_disconnected=False,
    )
    with pytest.warns(RuntimeWarning, match="cpnodes"):
        compute_geometry(grid, cp_geometry=True)
    assert not hasattr(grid.cells, "cpgeometry")
