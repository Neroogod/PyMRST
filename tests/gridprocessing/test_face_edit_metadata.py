"""Optional grid data must survive direct add_faces / remove_faces calls.

These are the low-level helpers behind makeInternalBoundary, PEBI construction
and the pinch/short-edge collapses. MRST's `addFaces`/`removeFaces` mutate the
grid in place, so every field they do not touch survives; the Python ports
rebuilt the grid and dropped them.
"""

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.private.add_faces import add_faces
from pymrst.gridprocessing.private.remove_faces import remove_faces


def _grid():
    """A 2x1x1 grid carrying every optional field the helpers may see."""
    grid = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
    grid.faces.tag = np.arange(grid.faces.num, dtype=np.int64)
    grid.cells.tags = np.arange(grid.cells.faces.size, dtype=np.int64)
    grid.faces.diameters = np.linspace(1.0, 2.0, grid.faces.num)
    grid.cells.diameters = np.array([1.5, 2.5])
    return grid


def _a_boundary_face(grid) -> int:
    return int(np.flatnonzero(np.any(grid.faces.neighbors == -1, axis=1))[0])


# --------------------------------------------------------------- remove_faces


def test_remove_faces_keeps_cart_dims_and_index_map():
    grid = _grid()
    reduced, _ = remove_faces(grid, np.array([_a_boundary_face(grid)]))
    assert reduced.cartDims == grid.cartDims
    assert np.array_equal(reduced.cells.indexMap, grid.cells.indexMap)


def test_remove_faces_subsets_every_per_face_array():
    grid = _grid()
    face = _a_boundary_face(grid)
    keep = np.setdiff1d(np.arange(grid.faces.num), [face])
    reduced, face_map = remove_faces(grid, np.array([face]))
    assert face_map[face] == -1
    for name in ("tag", "areas", "centroids", "normals", "diameters"):
        expected = np.asarray(getattr(grid.faces, name))[keep]
        actual = np.asarray(getattr(reduced.faces, name))
        assert actual.shape == expected.shape, name
        assert np.array_equal(actual, expected), name


def test_remove_faces_drops_cell_tags_with_their_half_faces():
    grid = _grid()
    face = _a_boundary_face(grid)
    dropped = grid.cells.tags[np.asarray(grid.cells.faces) == face]
    reduced, _ = remove_faces(grid, np.array([face]))
    assert reduced.cells.tags.size == reduced.cells.faces.size
    assert not np.isin(dropped, reduced.cells.tags).any()


def test_remove_faces_keeps_cell_arrays_addressable():
    grid = _grid()
    reduced, _ = remove_faces(grid, np.array([_a_boundary_face(grid)]))
    assert reduced.cells.volumes.size == reduced.cells.num
    assert reduced.cells.diameters.size == reduced.cells.num


# ------------------------------------------------------------------ add_faces


def test_add_faces_keeps_cart_dims_and_cell_arrays():
    grid = _grid()
    nodes = grid.face_nodes(_a_boundary_face(grid))
    extended, added = add_faces(
        grid, nodes, np.array([nodes.size]), np.array([[0, -1]])
    )
    assert extended.cartDims == grid.cartDims
    assert np.array_equal(extended.cells.indexMap, grid.cells.indexMap)
    assert extended.cells.volumes.size == extended.cells.num
    assert added.tolist() == [grid.faces.num]


def test_add_faces_zero_fills_the_new_face_tag():
    grid = _grid()
    nodes = grid.face_nodes(_a_boundary_face(grid))
    extended, _ = add_faces(
        grid, nodes, np.array([nodes.size]), np.array([[0, -1]])
    )
    assert extended.faces.tag.size == extended.faces.num
    assert np.array_equal(extended.faces.tag[:grid.faces.num], grid.faces.tag)
    assert extended.faces.tag[-1] == 0


def test_add_faces_invalidates_face_geometry_it_cannot_extend():
    grid = _grid()
    nodes = grid.face_nodes(_a_boundary_face(grid))
    extended, _ = add_faces(
        grid, nodes, np.array([nodes.size]), np.array([[0, -1]])
    )
    for name in ("areas", "normals", "centroids", "diameters"):
        assert getattr(extended.faces, name) is None, name
    recomputed = compute_geometry(extended)
    assert recomputed.faces.areas.size == extended.faces.num


def test_add_faces_places_cell_tags_beside_their_half_faces():
    grid = _grid()
    nodes = grid.face_nodes(_a_boundary_face(grid))
    extended, added = add_faces(
        grid, nodes, np.array([nodes.size]), np.array([[1, -1]]),
        tags=np.array([[7, 0]]),
    )
    assert extended.cells.tags.size == extended.cells.faces.size
    position = np.flatnonzero(np.asarray(extended.cells.faces) == added[0])
    assert position.size == 1
    assert extended.cells.tags[position[0]] == 7
    # untouched half-faces keep their original tags, in order
    others = np.asarray(extended.cells.faces) != added[0]
    assert np.array_equal(extended.cells.tags[others], grid.cells.tags)


def test_add_faces_defaults_cell_tags_to_zero_when_none_given():
    grid = _grid()
    nodes = grid.face_nodes(_a_boundary_face(grid))
    extended, added = add_faces(
        grid, nodes, np.array([nodes.size]), np.array([[1, -1]])
    )
    position = np.flatnonzero(np.asarray(extended.cells.faces) == added[0])
    assert extended.cells.tags[position[0]] == 0


def test_add_faces_rejects_mismatched_tag_counts():
    grid = _grid()
    nodes = grid.face_nodes(_a_boundary_face(grid))
    with pytest.raises(ValueError, match="one entry per new half-face"):
        add_faces(grid, nodes, np.array([nodes.size]), np.array([[1, -1]]),
                  tags=np.array([1, 2, 3]))


def test_grid_without_optional_fields_is_unaffected():
    """Absent fields must stay absent rather than materialise as zeros."""
    grid = cart_grid([2, 1, 1], [2.0, 1.0, 1.0])
    grid.faces.tag = None                                # cart_grid sets both
    grid.cells.tags = None
    nodes = grid.face_nodes(_a_boundary_face(grid))
    extended, _ = add_faces(
        grid, nodes, np.array([nodes.size]), np.array([[0, -1]])
    )
    assert extended.faces.tag is None
    assert extended.cells.tags is None
    reduced, _ = remove_faces(extended, np.array([extended.faces.num - 1]))
    assert reduced.faces.tag is None
    assert reduced.cells.tags is None


def test_add_then_remove_restores_the_original_metadata():
    grid = _grid()
    nodes = grid.face_nodes(_a_boundary_face(grid))
    extended, added = add_faces(
        grid, nodes, np.array([nodes.size]), np.array([[1, -1]]),
        tags=np.array([[7, 0]]),
    )
    restored, _ = remove_faces(extended, added)
    assert restored.faces.num == grid.faces.num
    assert np.array_equal(restored.faces.tag, grid.faces.tag)
    assert np.array_equal(restored.cells.tags, grid.cells.tags)
    assert np.array_equal(restored.cells.faces, grid.cells.faces)
