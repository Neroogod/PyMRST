"""Foundation and runnable DFN tests for the native ADFNE15 port."""

import numpy as np
import pytest

from pymrst.modules.adfne15 import (
    ADFNEMeshUnavailableError, ADFNETensorUnavailableError, angle,
    bounding_box, center, clip_lines,
    clip_polygons, cluster, filter_ids, flatri,
    generate_dfn, lengths, orientation, plane, pointline, polygon_normals,
    polyline, project, scale, unique_rows, mesh, naned, net, not_indices,
    distance, field, label, ray, relabel, reshape, rotate, stack, tensor,
    CData, GLOBALS, as_cell, as_struct, clock, convert, ellipse, geometry,
    get_shape_poly, inpoly, normalized, option, polygon_triangles, polyclip,
    setstructfields, sphere, to_2d, to_3d, translate, wrap,
)


def test_angle_and_scale_match_adfne_conventions():
    np.testing.assert_array_equal(angle([-10, 0, 90, 180, 370], 360),
                                  [350, 0, 90, 180, 10])
    np.testing.assert_array_equal(angle([-10, 45, 100, 190], 90),
                                  [10, 45, 80, 10])
    np.testing.assert_allclose(scale([0, .5, 1], 0, 1, 10, 20), [10, 15, 20])


def test_line_geometry_clipping_bbox_and_orientation():
    lines = np.array([[-1, .5, 2, .5], [.2, .2, .8, .8], [-2, -2, -1, -1]])
    clipped, mask = clip_lines(lines, return_mask=True)
    np.testing.assert_allclose(clipped, [[0, .5, 1, .5], [.2, .2, .8, .8]])
    np.testing.assert_array_equal(mask, [True, True, False])
    np.testing.assert_allclose(lengths(clipped), [1, np.sqrt(.72)])
    np.testing.assert_allclose(center(clipped)[0], [.5, .5])
    np.testing.assert_allclose(bounding_box(clipped), [0, .2, 1, .8])
    np.testing.assert_allclose(orientation(clipped).direction, [0, 45])


def test_polygon_plane_projection_clipping_and_normal():
    polygon = np.array([[-.5, .2, .2], [.5, .2, .2], [.5, .8, .8], [-.5, .8, .8]])
    definition = plane(polygon)
    flat = project(polygon, definition)
    assert flat.shape == (4, 2)
    clipped, mask = clip_polygons([polygon], return_mask=True)
    assert mask.tolist() == [True]
    assert np.all((clipped[0] >= -1e-14) & (clipped[0] <= 1+1e-14))
    assert np.linalg.norm(polygon_normals(clipped)[0]) > 0


def test_topology_helpers_preserve_stable_indices():
    points = np.array([[0, 0], [1, 0], [1, 1]])
    np.testing.assert_array_equal(polyline(points),
                                  [[0, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0]])
    np.testing.assert_array_equal(pointline(points, [[0, 2]]), [[0, 0, 1, 1]])
    values, first, inverse = unique_rows([[0, 0], [1, 1], [0, 0]], 1e-12,
                                         return_indices=True)
    np.testing.assert_array_equal(values, [[0, 0], [1, 1]])
    np.testing.assert_array_equal(first, [0, 1])
    np.testing.assert_array_equal(inverse, [0, 1, 0])
    assert [item.tolist() for item in cluster([[1, 2], [4, 5], [2, 3]])] == [
        [1, 2, 3], [4, 5]]


def test_seeded_2d_dfn_is_bounded_and_reproducible():
    first = generate_dfn(count=40, direction=35, direction_variation=0,
                         minimum_length=.1, maximum_length=.5, seed=9)
    second = generate_dfn(count=40, direction=35, direction_variation=0,
                          minimum_length=.1, maximum_length=.5, seed=9)
    np.testing.assert_array_equal(first.original, second.original)
    np.testing.assert_array_equal(first.fractures, second.fractures)
    assert first.fractures.shape[1] == 4
    assert np.all(first.fractures >= 0) and np.all(first.fractures <= 1)
    np.testing.assert_allclose(orientation(first.original).direction, 35)


def test_seeded_3d_dfn_produces_clipped_planar_polygons():
    result = generate_dfn(dimension=3, count=12, shape="square",
                          polygon_segments=8, minimum_length=.1,
                          maximum_length=.6, seed=4)
    assert len(result.original) == 12 and len(result.fractures) > 0
    for polygon in result.fractures:
        assert polygon.shape[1] == 3
        assert np.all(polygon >= -1e-12) and np.all(polygon <= 1+1e-12)
        assert np.linalg.matrix_rank(polygon-polygon[0], tol=1e-10) <= 2


def test_remaining_small_layout_owners_match_adfne_conventions():
    np.testing.assert_array_equal(net([1,2,3],[4,5]),
                                  [[1,4],[1,5],[2,4],[2,5],[3,4],[3,5]])
    np.testing.assert_array_equal(stack([np.array([[1,2]]),np.array([[3,4]])]),
                                  [[1,2],[3,4]])
    matrix=reshape(np.arange(1,17),4,4)
    np.testing.assert_array_equal(flatri(matrix),[5,9,13,10,14,15])
    np.testing.assert_array_equal(filter_ids([[1,2],[2,3],[2,3]]),[2,3])
    np.testing.assert_array_equal(not_indices([1,0,4,3,2,1],[0,2]),[0,3,2,1])
    coords=naned([[0,1,2,3]])
    np.testing.assert_allclose(coords[0][:2,0],[0,2]); assert np.isnan(coords[0][2,0])
    with pytest.raises(ADFNEMeshUnavailableError,match="not included"):
        mesh([[0,0,1,1]],.1)


def test_transform_distance_and_label_owners():
    points = np.array([[0., 0.], [1., 0.], [1., 1.]])
    np.testing.assert_allclose(translate(points, [2, 3]), points + [2, 3])
    np.testing.assert_allclose(rotate(points, angle=90, center=[0, 0]),
                               [[0, 0], [0, 1], [-1, 1]], atol=1e-15)
    np.testing.assert_allclose(ray(points),
                               [[2/3, 1/3, 0, 0], [2/3, 1/3, 1, 0],
                                [2/3, 1/3, 1, 1]])
    np.testing.assert_allclose(distance([[.5, .5], [2, .5]], [points]),
                               [0, 1])
    np.testing.assert_array_equal(label([[0, 2], [3]], 5), [1, -1, 1, 2, -2])
    np.testing.assert_array_equal(relabel([1, 1, -1, 2, 2, -2, 1, 2, 3]),
                                  [2, 2, -1, 3, 3, -2, 2, 3, 1])
    np.testing.assert_array_equal(wrap(np.arange(1, 11), 4),
                                  [1, 2, 3, 4, 1, 2, 3, 4, 1, 2])
    assert field({"Line": 4}, "Line") == 4
    with pytest.raises(ADFNETensorUnavailableError, match="not included"):
        tensor(None)


def test_polygon_domain_and_geometry_owners():
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    np.testing.assert_array_equal(inpoly([[.5, .5], [1, .5], [2, .5]], square),
                                  [True, True, False])
    square3 = np.column_stack((square, np.zeros(4)))
    np.testing.assert_array_equal(inpoly([[.5, .5, 0], [.5, .5, .1]], square3,
                                         tolerance=1e-8), [True, False])
    np.testing.assert_allclose(ellipse(center=[1, 2], radii=[2, 1], resolution=4),
                               [[3, 2], [1, 3], [-1, 2], [1, 1]], atol=1e-14)
    globe = sphere(center=[1, 2, 3], radius=2, resolution=6)
    assert globe.x.shape == globe.y.shape == globe.z.shape == (7, 7)
    assert geometry("sphere", resolution=6).x.shape == (7, 7)
    boundaries = get_shape_poly(radius=1, center=[0, 0, 0], length=1,
                                resolution=4)
    assert len(boundaries) == 6
    candidate = np.array([[.5, -2, -2], [.5, 2, -2], [.5, 2, 2],
                          [.5, -2, 2]])
    clipped, mask = polyclip([candidate], boundaries, return_mask=True)
    assert mask.tolist() == [True]
    np.testing.assert_allclose(np.sort(clipped[0][:, 1:], axis=0),
                               np.sort(np.array([[-1, 0], [0, -1], [1, 0],
                                                 [0, 1]]), axis=0), atol=1e-14)


def test_convert_option_globals_and_cdata_owners():
    assert clock(1000) == "00:16:40.00"
    np.testing.assert_allclose(convert("deg", np.pi), 180)
    np.testing.assert_allclose(convert("rad", 180), np.pi)
    np.testing.assert_allclose(normalized([3, 4]), [.6, .8])
    assert as_struct("a", 1, "b", 34) == {"a": 1, "b": 34}
    assert as_cell({"a": 1, "b": 2}) == ["a", 1, "b", 2]
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    triangles = polygon_triangles(square)[0]
    assert len(triangles) == 4 and all(value.shape == (3, 2) for value in triangles)
    np.testing.assert_array_equal(to_3d([[1, 2], [3, 4]]),
                                  [[1, 2, 0], [3, 4, 0]])
    points3 = np.column_stack((square, np.zeros(4)))
    assert to_2d(points3).shape == (4, 2)
    merged = setstructfields({"a": {"x": 1}, "b": 2},
                             {"a": {"y": 3}})
    assert merged == {"a": {"x": 1, "y": 3}, "b": 2}
    opts = option(("fc", "red", "r", 2), "fc", "blue", "lw", 1)
    assert opts["facecolor"] == "red" and opts["radius"] == 2
    assert GLOBALS.segment == 24 and GLOBALS.polygons["Left"].shape == (4, 3)
    np.testing.assert_array_equal(CData.almost([0, 1], 0, .1), [True, False])
    columns = CData.deal([[1, 2], [3, 4]])
    np.testing.assert_array_equal(columns[0], [1, 3])
    np.testing.assert_array_equal(CData.get([[1, 2], [3, 4]], 10), [3, 4])
    less, equal, greater = CData.leg([.4, .5, .6], .5, .01)
    np.testing.assert_array_equal(less, [True, False, False])
    np.testing.assert_array_equal(equal, [False, True, False])
    np.testing.assert_array_equal(greater, [False, False, True])
