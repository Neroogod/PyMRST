import numpy as np

from pymrst.gridprocessing import compute_geometry, glue_2d_grid
from pymrst.gridprocessing.tessellation_grid import tessellation_grid


def _square(x0):
    points = np.array([
        [x0, 0.0], [x0 + 1.0, 0.0],
        [x0 + 1.0, 1.0], [x0, 1.0],
    ])
    return tessellation_grid(points, [[0, 1, 2, 3]])


def test_glue_mapping_tracks_shared_nodes_faces_and_cells():
    first, second = _square(0.0), _square(1.0)
    glued, mapping = glue_2d_grid(first, second, return_mapping=True)
    glued = compute_geometry(glued)
    glued.check()
    assert np.array_equal(mapping.cells, [[0, -1], [-1, 0]])
    shared_nodes = np.all(mapping.nodes >= 0, axis=1)
    assert np.count_nonzero(shared_nodes) == 2
    shared_faces = np.all(mapping.faces >= 0, axis=1)
    assert np.count_nonzero(shared_faces) == 1
    face = np.flatnonzero(shared_faces).item()
    assert set(glued.faces.neighbors[face]) == {0, 1}


def test_glue_mapping_marks_nonconformal_split_faces_without_parent():
    coarse = _square(0.0)
    points = np.array([
        [1.0, 0.0], [2.0, 0.0], [2.0, 0.5],
        [2.0, 1.0], [1.0, 1.0], [1.0, 0.5],
    ])
    fine = tessellation_grid(points, [[0, 1, 2, 5], [5, 2, 3, 4]])
    glued, mapping = glue_2d_grid(coarse, fine, return_mapping=True)
    glued.check()
    assert glued.cells.num == 3
    coarse_pairs = coarse.faces.nodes.reshape(-1, 2)
    shared_coarse = next(
        face for face, pair in enumerate(coarse_pairs)
        if np.allclose(coarse.nodes.coords[pair, 0], 1.0)
    )
    assert not np.any(mapping.faces[:, 0] == shared_coarse)
    internal = glued.faces.neighbors[
        np.all(glued.faces.neighbors >= 0, axis=1)
    ]
    assert np.count_nonzero(np.any(internal == 0, axis=1)) == 2
