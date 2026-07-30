"""Focused invariants for MRST's private gridprocessing helpers.

These tests validate addFaces, copyFaces, emptyGrid, insertInPackedData,
removeFaces, removeFromPackedData, extractGridInfoFromPolygons, sortEdges,
and buildCornerPtNodes through their Python implementations or consolidated
public consumers.
"""

import numpy as np

from pymrst.gridprocessing import (
    add_faces,
    build_corner_pt_nodes,
    cart_grid,
    compute_grid_slice_polygons,
    empty_grid,
    remove_faces,
)
from pymrst.gridprocessing.make_internal_boundary import _copy_faces
from pymrst.gridprocessing.make_layered_grid import _sort_edges
from pymrst.gridprocessing.private.packed_data import (
    insert_in_packed_data,
    remove_from_packed_data,
)


def test_insert_and_remove_packed_data_exact_segments():
    """insertInPackedData and removeFromPackedData preserve segment layout."""
    pos = np.array([0, 2, 3, 6])
    data = np.array([10, 11, 20, 30, 31, 32])
    inserted, new_pos = insert_in_packed_data(
        pos, data, np.array([0, 2, 2]), np.array([12, 34, 33])
    )
    assert np.array_equal(new_pos, [0, 3, 4, 9])
    assert np.array_equal(inserted, [10, 11, 12, 20, 30, 31, 32, 33, 34])

    restored, restored_pos = remove_from_packed_data(
        new_pos, inserted, np.array([12, 33, 34])
    )
    assert np.array_equal(restored_pos, pos)
    assert np.array_equal(restored, data)


def test_empty_add_copy_remove_face_roundtrip():
    """emptyGrid, addFaces, copyFaces, and removeFaces compose correctly."""
    coords = np.array([
        [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0],
    ])
    grid = empty_grid(1, coords)
    face_nodes = np.array([0, 1, 1, 2, 2, 3, 3, 0])
    grid, added = add_faces(
        grid, face_nodes, np.full(4, 2), np.tile([0, -1], (4, 1))
    )
    copied_nodes, copied_counts, copied_neighbors = _copy_faces(grid, added[[3, 1]])
    assert np.array_equal(copied_counts, [2, 2])
    assert np.array_equal(copied_nodes, [3, 0, 1, 2])
    assert np.array_equal(copied_neighbors, [[0, -1], [0, -1]])

    reduced, face_map = remove_faces(grid, added[[1, 3]])
    assert np.array_equal(face_map, [0, -1, 1, -1])
    assert np.array_equal(reduced.cells.faces, [0, 1])
    assert np.array_equal(reduced.faces.nodes, [0, 1, 2, 3])


def test_sort_edges_reconstructs_oriented_ring():
    """sortEdges connects a scrambled set of oriented polygon edges."""
    edges = np.array([[2, 3], [0, 1], [3, 0], [1, 2]])
    ordered = _sort_edges(edges)
    assert np.array_equal(ordered[:-1, 1], ordered[1:, 0])
    assert ordered[-1, 1] == ordered[0, 0]
    assert {tuple(edge) for edge in ordered} == {tuple(edge) for edge in edges}


def test_extract_grid_info_from_polygons_consumer():
    """extractGridInfoFromPolygons produces ordered cell slice polygons."""
    grid = cart_grid([2, 1, 1], [2.0, 1.0, 1.0])
    polygons = compute_grid_slice_polygons(
        grid,
        np.array([[0.5, 0.0, 0.0], [0.5, 1.0, 0.0]]),
        cut_dir=np.array([0.0, 0.0, 1.0]),
    )
    assert np.array_equal(polygons["cell_ix"], [0])
    assert np.array_equal(np.diff(polygons["node_pos"]), [4])
    assert np.allclose(polygons["coords_3d"][:, 0], 0.5)


def test_build_corner_point_nodes_unit_cube():
    """buildCornerPtNodes interpolates the eight corners of a unit cube."""
    coord = []
    for j in range(2):
        for i in range(2):
            coord.extend([i, j, 0.0, i, j, 1.0])
    grdecl = {
        "cartDims": np.array([1, 1, 1]),
        "COORD": np.asarray(coord),
        "ZCORN": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
    }
    x, y, z = build_corner_pt_nodes(grdecl)
    assert x.shape == y.shape == z.shape == (2, 2, 2)
    assert np.array_equal(x[:, 0, 0], [0.0, 1.0])
    assert np.array_equal(y[0, :, 0], [0.0, 1.0])
    assert np.all(z[:, :, 0] == 0.0)
    assert np.all(z[:, :, 1] == 1.0)
