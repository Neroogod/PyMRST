import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry, remove_cells


def test_remove_cells_returns_original_entity_mappings():
    grid = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
    reduced, cells, faces, nodes = remove_cells(
        grid, np.array([False, True, False]), return_mapping=True
    )
    reduced.check()
    assert np.array_equal(cells, [0, 2])
    assert np.array_equal(reduced.cells.indexMap, grid.cells.indexMap[cells])
    assert np.array_equal(reduced.nodes.coords, grid.nodes.coords[nodes])
    assert reduced.faces.num == faces.size


def test_remove_cells_filters_and_renumbers_nncs():
    grid = cart_grid([4, 1, 1], [4.0, 1.0, 1.0])
    grid.nnc = {
        "cells": np.array([[0, 2], [1, 3], [0, 3]]),
        "trans": np.array([10.0, 20.0, 30.0]),
    }
    reduced = remove_cells(grid, [1])
    assert np.array_equal(reduced.nnc["cells"], [[0, 1], [0, 2]])
    assert np.array_equal(reduced.nnc["trans"], [10.0, 30.0])


def test_remove_cells_remaps_preserved_corner_point_nodes():
    grid = cart_grid([2, 1, 1], [2.0, 1.0, 1.0])
    def nodes_of_cell(cell):
        return np.unique(np.concatenate([
            grid.face_nodes(face) for face in grid.cell_faces(cell)
        ]))

    grid.cells.cpnodes = np.array([nodes_of_cell(0), nodes_of_cell(1)])
    reduced, _, _, nodes = remove_cells(grid, [1], return_mapping=True)
    old_to_new = np.full(grid.nodes.num, -1, dtype=np.int64)
    old_to_new[nodes] = np.arange(nodes.size)
    assert np.array_equal(reduced.cells.cpnodes,
                          old_to_new[grid.cells.cpnodes[[0]]])


def test_remove_cells_rejects_wrong_length_logical_mask():
    grid = cart_grid([2, 1, 1])
    with np.testing.assert_raises(ValueError):
        remove_cells(grid, np.array([True]))
