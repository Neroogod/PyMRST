"""computeGeometry's operational options: MaxBlockSize and verbose.

Neither affects the answer. `max_block_size` bounds how many cells the 3D cell
pass holds at once - the pass that allocates one row per cell-face triangle -
and `verbose` reports progress. The tests that matter are the ones proving the
numbers do not move.
"""

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, tensor_grid
from pymrst.gridprocessing.compute_geometry import (
    MAX_BLOCK_SIZE,
    _partition_cells,
    compute_geometry,
)


def _perturbed(dims=(5, 4, 3), seed: int = 7):
    """A grid whose interior nodes are jittered, so no two cells are alike."""
    grid = cart_grid(list(dims), [float(d) for d in dims])
    points = grid.nodes.coords
    low, high = points.min(axis=0), points.max(axis=0)
    inner = np.all((points > low + 1e-9) & (points < high - 1e-9), axis=1)
    rng = np.random.default_rng(seed)
    points[inner] += 0.15 * (rng.random((int(inner.sum()), 3)) - 0.5)
    return points


def _with(points, **kwargs):
    grid = cart_grid([5, 4, 3], [5.0, 4.0, 3.0])
    grid.nodes.coords[:] = points
    return compute_geometry(grid, **kwargs)


# ------------------------------------------------------------------ partition


@pytest.mark.parametrize("num_cells", [1, 2, 7, 10, 100, 997])
@pytest.mark.parametrize("size", [1, 2, 3, 7, 10, 1000])
def test_blocks_tile_the_cells_exactly(num_cells, size):
    blocks = _partition_cells(num_cells, size)
    assert blocks[0][0] == 0 and blocks[-1][1] == num_cells
    assert all(a[1] == b[0] for a, b in zip(blocks, blocks[1:]))


@pytest.mark.parametrize("num_cells", [7, 10, 100, 997])
@pytest.mark.parametrize("size", [2, 3, 7, 64])
def test_blocks_are_balanced_and_within_the_requested_size(num_cells, size):
    sizes = [hi - lo for lo, hi in _partition_cells(num_cells, size)]
    assert max(sizes) - min(sizes) <= 1, "MRST evens the split out"
    assert max(sizes) <= size


@pytest.mark.parametrize("size", [None, -1, -20])
def test_empty_or_negative_size_means_a_single_pass(size):
    assert _partition_cells(50, size) == [(0, 50)]


def test_size_larger_than_the_grid_means_a_single_pass():
    assert _partition_cells(50, 10_000) == [(0, 50)]


def test_no_cells_needs_no_blocks():
    assert _partition_cells(0, 10) == []


def test_zero_block_size_is_rejected():
    with pytest.raises(ValueError, match="positive"):
        _partition_cells(10, 0)


# ------------------------------------------------------------- invariance


@pytest.mark.parametrize("size", [1, 2, 3, 7, 13, 59, 60, 61, None, -1])
def test_block_size_does_not_change_any_number(size):
    points = _perturbed()
    reference = _with(points, max_block_size=None)
    blocked = _with(points, max_block_size=size)
    assert np.array_equal(blocked.cells.volumes, reference.cells.volumes)
    assert np.array_equal(blocked.cells.centroids, reference.cells.centroids)
    assert np.array_equal(blocked.faces.areas, reference.faces.areas)
    assert np.array_equal(blocked.faces.normals, reference.faces.normals)
    assert np.array_equal(blocked.faces.centroids, reference.faces.centroids)


def test_default_block_size_runs_small_grids_in_one_pass():
    assert _partition_cells(60, MAX_BLOCK_SIZE) == [(0, 60)]


def test_blocking_holds_for_unequal_tensor_spacing():
    axes = (np.array([0.0, 1.0, 2.5, 5.0, 10.0]),
            np.array([0.0, 2.0, 3.0, 7.0]),
            np.array([0.0, 1.0, 4.0]))
    single = compute_geometry(tensor_grid(*axes), max_block_size=None)
    blocked = compute_geometry(tensor_grid(*axes), max_block_size=5)
    assert np.array_equal(single.cells.volumes, blocked.cells.volumes)
    assert np.array_equal(single.cells.centroids, blocked.cells.centroids)


def test_single_cell_block_still_conserves_total_volume():
    grid = compute_geometry(cart_grid([4, 4, 4], [1.0, 1.0, 1.0]),
                            max_block_size=1)
    assert np.isclose(grid.cells.volumes.sum(), 1.0)
    assert np.all(grid.cells.volumes > 0)


def test_blocking_leaves_hinge_nodes_working():
    """The two options are independent: hinge centers still take effect."""
    grid = cart_grid([2, 2, 2], [2.0, 2.0, 2.0])
    face = int(grid.cells.faces[4])
    hinge = grid.nodes.coords[grid.face_nodes(face)].mean(axis=0) + [0.1, 0.0, 0.2]
    hinges = {"faces": [face], "nodes": [hinge]}
    single = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]),
                              hinge_nodes=hinges, max_block_size=None)
    blocked = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]),
                               hinge_nodes=hinges, max_block_size=2)
    assert np.array_equal(single.cells.volumes, blocked.cells.volumes)
    assert np.array_equal(single.faces.areas, blocked.faces.areas)


# ----------------------------------------------------------------- verbose


def test_verbose_reports_every_block(capsys):
    compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]),
                     max_block_size=2, verbose=True)
    out = capsys.readouterr().out
    assert out.count("Processing Cells") == 4          # 8 cells in blocks of 2
    assert "cells/second" in out
    assert "Total 3D Geometry Processing Time" in out


def test_quiet_by_default(capsys):
    compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))
    assert capsys.readouterr().out == ""


def test_verbose_reports_the_two_2d_phases(capsys):
    compute_geometry(cart_grid([3, 3], [3.0, 3.0]), verbose=True)
    out = capsys.readouterr().out
    assert "Computing normals, areas, and centroids" in out
    assert "Computing cell volumes and centroids" in out
