"""MRST-owner layout tests for Intersect/Snap/Group/Sort/Split/Isolated."""

import numpy as np

from pymrst.modules.adfne15 import (group, intersect, isolated, snap,
                                    sort_points, split)


NETWORK = np.array([
    [0.0, .5, 1.0, .5],
    [.25, 0.0, .25, 1.0],
    [.75, 0.0, .75, 1.0],
    [.1, .1, .2, .1],
])


def test_intersect_returns_geometry_pairs_and_cluster_labels():
    result = intersect(NETWORK)
    np.testing.assert_allclose(result.geometry, [[.25, .5], [.75, .5]])
    np.testing.assert_array_equal(result.pairs, [[0, 1], [0, 2]])
    np.testing.assert_array_equal(result.labels, [1, 1, 1, -1])


def test_snap_returns_all_distances_or_first_matching_projection():
    lines = np.array([[0, 0, 1, 0], [0, 1, 1, 1]])
    points = np.array([[.2, .01], [.8, .9]])
    distances = snap(points, lines, 0, distances_only=True)
    np.testing.assert_allclose(distances, [[.01, .99], [.9, .1]])
    snapped, mask = snap(points, lines, .11)
    np.testing.assert_allclose(snapped, [[.2, 0], [.8, 1]])
    np.testing.assert_array_equal(mask, [[True, False], [False, True]])


def test_group_sort_and_split_preserve_source_lineage():
    intersection = intersect(NETWORK)
    points, neighbours = group(intersection.geometry, intersection.pairs, 4)
    np.testing.assert_allclose(points[0], [[.25, .5], [.75, .5]])
    np.testing.assert_array_equal(neighbours[0], [1, 2])
    ordered = sort_points([[.75, .5], [0, .5], [.25, .5], [1, .5]], NETWORK[0])
    np.testing.assert_allclose(ordered[:, 0], [0, .25, .75, 1])
    result = split(NETWORK)
    np.testing.assert_array_equal(result.segment_counts, [3, 2, 2, 1])
    np.testing.assert_array_equal(np.bincount(result.source_ids), [3, 2, 2, 1])
    assert result.lines.shape == (8, 4)


def test_interval_split_and_isolation_cover_2d_and_3d():
    result = split([[0, 0, 1, 0]], interval=.3)
    assert len(result.lines[0]) == 5
    connected = np.array([[0, 0, 1, 0], [1, 0, 2, 0], [2, 0, 0, 0]])
    np.testing.assert_array_equal(isolated(connected), [False, False, False])
    lines3d = np.array([[0, 0, 0, 1, 0, 0], [1, 0, 0, 2, 0, 0]])
    np.testing.assert_array_equal(isolated(lines3d), [True, True])


def test_convex_3d_polygon_intersection_returns_shared_line():
    horizontal = np.array([[0, 0, .5], [1, 0, .5], [1, 1, .5], [0, 1, .5]])
    vertical = np.array([[.5, 0, 0], [.5, 1, 0], [.5, 1, 1], [.5, 0, 1]])
    result = intersect([horizontal, vertical])
    assert result.geometry.shape == (1, 6)
    endpoints = result.geometry.reshape(2, 3)
    np.testing.assert_allclose(endpoints[:, 0], .5)
    np.testing.assert_allclose(endpoints[:, 2], .5)
    np.testing.assert_allclose(np.sort(endpoints[:, 1]), [0, 1])
