import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.params import fluxside, pside
from pymrst.utils.gridtools import boundary_face_indices


@pytest.fixture
def grid():
    return compute_geometry(cart_grid([4, 3, 2], [40.0, 30.0, 20.0]))


def test_empty_in_plane_ranges_mean_full_side(grid):
    full = boundary_face_indices(grid, "xmin")
    empty = boundary_face_indices(grid, "west", [], [])
    np.testing.assert_array_equal(empty, full)


def test_side_aliases_and_ranges_append_without_overlap(grid):
    west = pside(None, grid, "Left", 2.0, i1=[0, 2], i2=[0])
    east = pside(west, grid, "RIGHT", 1.0, i1=[1], i2=[1])
    assert east.face.size == 3
    assert east.type == ["pressure"]*3


def test_perpendicular_range_can_exclude_regular_outer_side(grid):
    with pytest.warns(RuntimeWarning, match="Empty list"):
        result = pside(None, grid, "xmin", 1.0, search_range=[1])
    assert result is None
    selected = pside(None, grid, "xmin", 1.0, search_range=[0])
    assert selected.face.size == 6


def test_flux_vector_is_area_weighted_like_mrst(grid):
    faces = boundary_face_indices(grid, "xmin")
    raw = np.arange(1, faces.size + 1, dtype=float)
    bc = fluxside(None, grid, "xmin", raw)
    expected = raw*grid.faces.areas[faces]/grid.faces.areas[faces].sum()
    np.testing.assert_allclose(bc.value, expected)


def test_flux_scalar_preserves_requested_total(grid):
    bc = fluxside(None, grid, "ymax", -3.5, sat=[.2, .8])
    assert np.isclose(bc.value.sum(), -3.5)
    np.testing.assert_allclose(bc.sat, np.tile([.2, .8], (8, 1)))


def test_side_helpers_validate_ranges_flux_shapes_and_dimensions(grid):
    with pytest.raises(IndexError, match="i1"):
        pside(None, grid, "xmin", 1.0, i1=[3])
    with pytest.raises(IndexError, match="range"):
        pside(None, grid, "xmin", 1.0, search_range=[4])
    with pytest.raises(ValueError, match="scalar"):
        fluxside(None, grid, "xmin", [1.0, 2.0])
    grid2 = compute_geometry(cart_grid([3, 2], [30.0, 20.0]))
    with pytest.raises(ValueError, match="unsupported"):
        pside(None, grid2, "zmin", 1.0)
