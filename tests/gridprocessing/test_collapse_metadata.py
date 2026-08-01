"""Metadata handling across the aggressive collapse operations.

`remove_pinch` and `remove_short_edges` merge nodes and drop faces and cells.
That leaves three things to get right: index arrays into the node table must
follow the merge, dynamically attached data (`cells.cpnodes`, `g.nnc`) must
survive the rebuild, and geometry computed before the collapse must not be
handed back looking valid.
"""

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.remove_pinch import remove_pinch
from pymrst.gridprocessing.remove_short_edges import remove_short_edges

COLLAPSE_OPS = [
    pytest.param(lambda g: remove_pinch(g), id="remove_pinch"),
    pytest.param(lambda g: remove_short_edges(g, tol=1e-9), id="remove_short_edges"),
]


def _pinched_column(with_metadata: bool = True):
    """1x1x2 column whose lower layer is collapsed to zero thickness."""
    grid = cart_grid([1, 1, 2], [1.0, 1.0, 2.0])
    coords = grid.nodes.coords
    coords[np.isclose(coords[:, 2], 1.0), 2] = 0.0
    with warnings.catch_warnings():                      # degenerate faces
        warnings.simplefilter("ignore")
        grid = compute_geometry(grid)
    if with_metadata:
        grid.cells.cpnodes = np.array([np.arange(8), np.arange(4, 12)])
        grid.nnc = SimpleNamespace(cells=np.array([[0, 1]]), trans=np.array([2.5]))
    return grid


def _run(collapse, grid):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return collapse(grid)


@pytest.mark.parametrize("collapse", COLLAPSE_OPS)
def test_collapse_actually_removes_the_pinched_cell(collapse):
    grid = _pinched_column()
    nodes_in = grid.nodes.num                            # both ops mutate `grid`
    out = _run(collapse, grid)
    assert out.cells.num == 1
    assert out.nodes.num < nodes_in


@pytest.mark.parametrize("collapse", COLLAPSE_OPS)
def test_corner_point_nodes_survive_and_stay_in_range(collapse):
    out = _run(collapse, _pinched_column())
    cpnodes = out.cells.cpnodes
    assert cpnodes is not None
    assert cpnodes.shape[0] == out.cells.num
    assert cpnodes.min() >= 0 and cpnodes.max() < out.nodes.num


@pytest.mark.parametrize("collapse", COLLAPSE_OPS)
def test_corner_point_nodes_still_address_the_right_corners(collapse):
    """In range is not enough - they must land on the same coordinates."""
    grid = _pinched_column()
    original = {tuple(np.round(p, 12)) for p in grid.nodes.coords}
    out = _run(collapse, grid)
    corners = {tuple(np.round(p, 12))
               for p in out.nodes.coords[out.cells.cpnodes[0]]}
    assert corners <= original
    # the surviving cell spans the full column, so both z extremes appear
    assert {round(p[2], 12) for p in corners} == {0.0, 2.0}


@pytest.mark.parametrize("collapse", COLLAPSE_OPS)
def test_nnc_survives_and_drops_pairs_touching_removed_cells(collapse):
    out = _run(collapse, _pinched_column())
    assert hasattr(out, "nnc"), "nnc must not be dropped by the rebuild"
    assert np.asarray(out.nnc.cells).shape[0] == 0
    assert np.asarray(out.nnc.trans).size == 0


@pytest.mark.parametrize("collapse", COLLAPSE_OPS)
def test_stale_geometry_is_cleared_rather_than_returned(collapse):
    grid = _pinched_column()
    assert grid.faces.areas is not None                  # geometry on the way in
    out = _run(collapse, grid)
    assert out.faces.areas is None
    assert out.faces.normals is None
    assert out.faces.centroids is None
    assert out.cells.volumes is None
    assert out.cells.centroids is None


@pytest.mark.parametrize("collapse", COLLAPSE_OPS)
def test_geometry_recomputes_to_the_collapsed_shape(collapse):
    out = compute_geometry(_run(collapse, _pinched_column()))
    assert np.isclose(out.cells.volumes.sum(), 2.0)      # 1 x 1 x 2 column
    assert np.all(out.cells.volumes > 0)


@pytest.mark.parametrize("collapse", COLLAPSE_OPS)
def test_grid_without_metadata_is_untouched_by_the_new_handling(collapse):
    out = _run(collapse, _pinched_column(with_metadata=False))
    assert not hasattr(out.cells, "cpnodes")
    assert not hasattr(out, "nnc")


@pytest.mark.parametrize("collapse", COLLAPSE_OPS)
def test_unchanged_grid_keeps_its_geometry(collapse):
    """Nothing to collapse means nothing to invalidate."""
    grid = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
    areas = grid.faces.areas.copy()
    out = _run(collapse, grid)
    assert out.cells.num == 2
    assert out.faces.areas is not None
    assert np.array_equal(out.faces.areas, areas)


@pytest.mark.parametrize("collapse", COLLAPSE_OPS)
def test_custom_fields_ride_along(collapse):
    grid = _pinched_column()
    grid.provenance = "deck-A"
    out = _run(collapse, grid)
    assert getattr(out, "provenance", None) == "deck-A"
