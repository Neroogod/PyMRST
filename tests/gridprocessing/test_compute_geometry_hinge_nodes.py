from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry


def _warped_cube():
    grid = cart_grid([1, 1, 1])
    face = int(grid.cells.faces[4])
    nodes = grid.face_nodes(face)
    grid.nodes.coords[nodes[-1], 2] += 0.4
    hinge = grid.nodes.coords[nodes].mean(axis=0) + [0.1, -0.05, 0.2]
    return grid, face, nodes, hinge


def test_hinge_node_controls_face_triangulation_and_cell_geometry():
    source, face, nodes, hinge = _warped_cube()
    default = compute_geometry(deepcopy(source))
    hinged = compute_geometry(
        deepcopy(source), hinge_nodes={"faces": [face], "nodes": [hinge]}
    )

    coordinates = source.nodes.coords[nodes]
    following = np.roll(coordinates, -1, axis=0)
    sub_normals = np.cross(following - coordinates, hinge - coordinates) / 2
    sub_areas = np.linalg.norm(sub_normals, axis=1)
    sub_centroids = (coordinates + following + hinge) / 3
    expected_centroid = np.sum(sub_areas[:, None] * sub_centroids, axis=0) / np.sum(sub_areas)

    np.testing.assert_allclose(hinged.faces.normals[face], sub_normals.sum(axis=0))
    np.testing.assert_allclose(hinged.faces.areas[face], sub_areas.sum())
    np.testing.assert_allclose(hinged.faces.centroids[face], expected_centroid)
    assert not np.allclose(hinged.faces.areas[face], default.faces.areas[face])
    assert not np.allclose(hinged.cells.volumes, default.cells.volumes)
    assert not np.allclose(hinged.cells.centroids, default.cells.centroids)


def test_hinge_node_accepts_mrst_style_object_and_mean_is_default():
    source, face, nodes, _ = _warped_cube()
    center = source.nodes.coords[nodes].mean(axis=0)
    expected = compute_geometry(deepcopy(source))
    actual = compute_geometry(
        deepcopy(source),
        hinge_nodes=SimpleNamespace(faces=np.array([face]), nodes=center),
    )
    np.testing.assert_allclose(actual.faces.areas, expected.faces.areas)
    np.testing.assert_allclose(actual.faces.normals, expected.faces.normals)
    np.testing.assert_allclose(actual.faces.centroids, expected.faces.centroids)
    np.testing.assert_allclose(actual.cells.volumes, expected.cells.volumes)
    np.testing.assert_allclose(actual.cells.centroids, expected.cells.centroids)


def test_empty_hinge_node_structure_is_a_noop():
    expected = compute_geometry(cart_grid([1, 1, 1]))
    actual = compute_geometry(
        cart_grid([1, 1, 1]), hinge_nodes={"faces": [], "nodes": []}
    )
    np.testing.assert_allclose(actual.faces.areas, expected.faces.areas)
    np.testing.assert_allclose(actual.cells.volumes, expected.cells.volumes)


@pytest.mark.parametrize(
    "hinges, message",
    [
        ({"faces": [99], "nodes": [[0, 0, 0]]}, "out-of-range"),
        ({"faces": [0, 1], "nodes": [[0, 0, 0]]}, "shape"),
        ({"faces": [0.5], "nodes": [[0, 0, 0]]}, "integer"),
        ({"faces": [0], "nodes": [[0, np.nan, 0]]}, "finite"),
        ({"faces": [0]}, "faces and nodes"),
    ],
)
def test_hinge_node_validation(hinges, message):
    with pytest.raises(ValueError, match=message):
        compute_geometry(cart_grid([1, 1, 1]), hinge_nodes=hinges)


def test_hinge_nodes_rejected_for_non_3d_grid():
    with pytest.raises(ValueError, match="only supported for 3D"):
        compute_geometry(
            cart_grid([1, 1]),
            hinge_nodes={"faces": [0], "nodes": [[0, 0, 0]]},
        )
