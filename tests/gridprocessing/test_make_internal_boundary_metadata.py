from types import SimpleNamespace

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry, make_internal_boundary


def _half_face_tags(grid):
    return {
        (int(cell), int(face)): int(value)
        for cell, face, value in zip(
            grid.cell_no(), grid.cells.faces, grid.cells.tags
        )
    }


def test_make_internal_boundary_preserves_half_face_tags_and_sets_face_tag():
    grid = cart_grid([2, 1, 1], [2.0, 1.0, 1.0])
    internal = np.flatnonzero(np.all(grid.faces.neighbors >= 0, axis=1))
    original_face = int(internal[0])
    original_neighbors = grid.faces.neighbors[original_face].copy()
    original_tags = _half_face_tags(grid)
    grid.faces.tag = np.arange(grid.faces.num, dtype=np.int64) + 10

    boundary, replacements = make_internal_boundary(
        grid, [original_face], tag=77
    )
    boundary.check()
    pair = replacements[0].astype(np.int64)
    assert np.array_equal(boundary.faces.tag[pair], [77, 77])
    for new_face, cell in zip(pair, original_neighbors):
        assert _half_face_tags(boundary)[(int(cell), int(new_face))] == \
            original_tags[(int(cell), original_face)]
    retained_tags = np.delete(np.arange(grid.faces.num) + 10, original_face)
    assert np.array_equal(boundary.faces.tag[:retained_tags.size], retained_tags)


def test_make_internal_boundary_duplicate_and_boundary_inputs_keep_mapping():
    grid = cart_grid([2, 1, 1], [2.0, 1.0, 1.0])
    internal = int(np.flatnonzero(np.all(grid.faces.neighbors >= 0, axis=1))[0])
    exterior = int(np.flatnonzero(np.any(grid.faces.neighbors < 0, axis=1))[0])
    boundary, replacements = make_internal_boundary(
        grid, [internal, exterior, internal], check=True
    )
    boundary.check()
    assert np.array_equal(replacements[0], replacements[2])
    assert np.all(np.isnan(replacements[1]))


def test_make_internal_boundary_preserves_grid_and_cell_metadata():
    grid = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
    volumes = grid.cells.volumes.copy()
    centroids = grid.cells.centroids.copy()
    grid.nnc = SimpleNamespace(cells=np.array([[0, 1]]), trans=np.array([2.0]))
    internal = np.flatnonzero(np.all(grid.faces.neighbors >= 0, axis=1))
    boundary, _ = make_internal_boundary(grid, internal)
    boundary.check()
    assert boundary.cartDims == (2, 1, 1)
    assert np.array_equal(boundary.cells.volumes, volumes)
    assert np.array_equal(boundary.cells.centroids, centroids)
    assert np.array_equal(boundary.nnc.cells, [[0, 1]])
    assert np.array_equal(boundary.nnc.trans, [2.0])
    assert boundary.faces.areas is None
