import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry, extract_subgrid


def _nodes_of_cell(grid, cell):
    return np.unique(np.concatenate([
        grid.face_nodes(face) for face in grid.cell_faces(cell)
    ]))


def test_extract_subgrid_returns_global_maps_and_fields():
    grid = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
    sub, cells, faces, nodes = extract_subgrid(
        grid, np.array([True, False, True]), return_mapping=True
    )
    sub.check()
    assert np.array_equal(cells, [0, 2])
    assert np.array_equal(sub.cells.global_, cells)
    assert np.array_equal(sub.faces.global_, faces)
    assert np.array_equal(sub.nodes.global_, nodes)
    assert np.array_equal(getattr(sub.cells, "global"), cells)
    assert np.array_equal(sub.nodes.coords, grid.nodes.coords[nodes])


def test_extract_subgrid_remaps_corner_nodes_and_nncs():
    grid = cart_grid([4, 1, 1], [4.0, 1.0, 1.0])
    grid.cells.cpnodes = np.array([
        _nodes_of_cell(grid, cell) for cell in range(grid.cells.num)
    ])
    grid.nnc = {
        "cells": np.array([[0, 2], [1, 3], [0, 3]]),
        "trans": np.array([10.0, 20.0, 30.0]),
    }
    sub = extract_subgrid(grid, [0, 2, 3])
    assert np.array_equal(sub.nnc["cells"], [[0, 1], [0, 2]])
    assert np.array_equal(sub.nnc["trans"], [10.0, 30.0])
    assert sub.cells.cpnodes.shape == (3, 8)
    assert sub.cells.cpnodes.min() >= 0
    assert sub.cells.cpnodes.max() < sub.nodes.num


def test_extract_subgrid_sorts_and_deduplicates_requested_cells():
    grid = cart_grid([4, 1, 1])
    sub, cells, _, _ = extract_subgrid(
        grid, [3, 1, 3], return_mapping=True
    )
    assert np.array_equal(cells, [1, 3])
    assert np.array_equal(sub.cells.indexMap, grid.cells.indexMap[[1, 3]])


def test_extract_subgrid_rejects_wrong_length_logical_mask():
    grid = cart_grid([2, 1, 1])
    with np.testing.assert_raises(ValueError):
        extract_subgrid(grid, np.array([True]))
