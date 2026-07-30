from types import SimpleNamespace

import numpy as np

from pymrst.modules.ad_core.utils import compute_cp_geometry


def _two_cell_deck(active=(1, 1)):
    # Two unit cubes along x, expressed as a corner-point deck.
    coord = []
    for y in (0.0, 1.0):
        for x in (0.0, 1.0, 2.0):
            coord.extend((x, y, 0.0, x, y, 1.0))
    return {
        "cartDims": np.array([2, 1, 1]),
        "COORD": np.asarray(coord),
        "ZCORN": np.asarray([0.0] * 8 + [1.0] * 8),
        "ACTNUM": np.asarray(active),
    }


def test_all_corner_point_cells_have_centers_faces_and_dimensions():
    deck = _two_cell_deck()
    centers, face_centers, dims = compute_cp_geometry(
        None, deck, on_all_cpc=True
    )
    assert np.allclose(centers, [[0.5, 0.5, 0.5], [1.5, 0.5, 0.5]])
    assert face_centers.shape == (12, 3)
    assert np.allclose(
        face_centers[:6],
        [
            [0.0, 0.5, 0.5],
            [1.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 0.0],
            [0.5, 0.5, 1.0],
        ],
    )
    assert np.allclose(dims, np.ones((2, 3)))


def test_processed_grid_path_uses_active_index_map_and_half_face_tags():
    deck = _two_cell_deck(active=(0, 1))
    cells = SimpleNamespace(
        facePos=np.array([0, 6]),
        faces=np.arange(6),
        tags=np.arange(6),
        indexMap=np.array([1]),
    )
    grid = SimpleNamespace(cells=cells)
    centers, face_centers, dims = compute_cp_geometry(grid, deck)
    assert np.allclose(centers, [[1.5, 0.5, 0.5]])
    assert np.allclose(
        face_centers,
        [
            [1.0, 0.5, 0.5],
            [2.0, 0.5, 0.5],
            [1.5, 0.0, 0.5],
            [1.5, 1.0, 0.5],
            [1.5, 0.5, 0.0],
            [1.5, 0.5, 1.0],
        ],
    )
    assert np.allclose(dims, np.ones((2, 3)))


def test_matlab_style_faces_tags_are_accepted():
    cells = SimpleNamespace(
        facePos=np.array([0, 6]),
        indexMap=np.array([0]),
        faces=np.column_stack([np.arange(6), np.arange(1, 7)]),
    )
    grid = SimpleNamespace(cells=cells)
    _, face_centers, _ = compute_cp_geometry(grid, _two_cell_deck())
    assert np.allclose(face_centers[0], [0.0, 0.5, 0.5])
    assert np.allclose(face_centers[-1], [0.5, 0.5, 1.0])
