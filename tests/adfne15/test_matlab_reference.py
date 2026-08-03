"""Direct ADFNE 1.5 MATLAB fixture checks."""

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from pymrst.modules.adfne15 import (ADFNEMeshUnavailableError, GraphEdge,
                                    GraphModel, GraphNode, angle,
                                    boundary, bounding_box, center, clip_lines,
                                    cluster, filter_ids, flatri, intersect,
                                    isolated, lengths, mesh, naned, net,
                                    not_indices, reshape, stack,
                                    orientation, pipe, pointline, polyline,
                                    scale, snap, solve, split)


FIXTURE = Path(__file__).parents[2] / "scripts/matlab/adfne15/ref_foundation.mat"


@pytest.mark.skipif(not FIXTURE.exists(), reason="generate the direct MATLAB fixture")
def test_foundation_matches_direct_adfne15_fixture():
    reference = loadmat(FIXTURE, squeeze_me=True)
    lines = np.array([[-1, .5, 2, .5], [.2, .2, .8, .8], [-2, -2, -1, -1]])
    clipped, mask = clip_lines(lines, return_mask=True)
    np.testing.assert_array_equal(angle([-10, 0, 90, 180, 370], 360), reference["angles360"])
    np.testing.assert_array_equal(angle([-10, 45, 100, 190], 90), reference["angles90"])
    np.testing.assert_allclose(scale([0, .5, 1], 0, 1, 10, 20), reference["scaled"])
    np.testing.assert_allclose(clipped, reference["clipped"])
    np.testing.assert_array_equal(mask, reference["clip_mask"])
    np.testing.assert_allclose(lengths(clipped), reference["line_lengths"])
    np.testing.assert_allclose(center(clipped), reference["line_centers"])
    np.testing.assert_allclose(bounding_box(clipped), reference["line_bbox"])
    np.testing.assert_allclose(orientation(clipped).direction,
                               reference["line_orientations"])
    points = np.array([[0, 0], [1, 0], [1, 1]])
    np.testing.assert_array_equal(polyline(points), reference["poly_lines"])
    np.testing.assert_array_equal(pointline(points, [[0, 2]]).reshape(-1),
                                  np.asarray(reference["point_line"]).reshape(-1))
    expected_clusters = [np.asarray(value).reshape(-1).astype(int)-1
                         for value in reference["clusters"]]
    assert [value.tolist() for value in cluster([[0, 1], [3, 4], [1, 2]])] == [
        value.tolist() for value in expected_clusters]
    network = np.array([[0, .5, 1, .5], [.25, 0, .25, 1],
                        [.75, 0, .75, 1], [.1, .1, .2, .1]])
    intersections = intersect(network)
    np.testing.assert_allclose(intersections.geometry, reference["network_xts"])
    np.testing.assert_array_equal(intersections.pairs+1, reference["network_ids"])
    np.testing.assert_array_equal(intersections.labels, reference["network_labels"])
    snap_points = np.array([[.2, .01], [.8, .9]])
    snap_lines = np.array([[0, 0, 1, 0], [0, 1, 1, 1]])
    np.testing.assert_allclose(snap(snap_points, snap_lines, 0, distances_only=True),
                               reference["snap_distances"])
    snapped, snap_mask = snap(snap_points, snap_lines, .11)
    np.testing.assert_allclose(snapped, reference["snapped_points"])
    np.testing.assert_array_equal(snap_mask, reference["snap_mask"])
    np.testing.assert_array_equal(isolated([[0, 0, 1, 0], [1, 0, 2, 0],
                                            [2, 0, 0, 0]]),
                                  reference["isolated_mask"])
    flow_be=np.array([[0,0,0,0,1,0],[1,0,0,1,1,0]],float)
    np.testing.assert_allclose(boundary([[0,.25,0],[.5,.5,0],[1,.75,0]],flow_be,[1,0]),reference["boundary_values"])
    nodes=(GraphNode((0,),(1,),np.array([0,.5,0]),1),GraphNode((0,1),(0,2),np.array([.5,.5,0]),0),GraphNode((1,),(1,),np.array([1,.5,0]),2))
    edges=(GraphEdge((0,1),0,1,.5,1e-3,0,0),GraphEdge((1,2),0,1,.5,1e-3,0,0))
    flow=solve(GraphModel(nodes,edges,flow_be),[1,0])
    np.testing.assert_allclose([n.pressure for n in flow.nodes],reference["flow_pressures"])
    np.testing.assert_allclose([e.pressure_drop for e in flow.edges],reference["flow_drops"])
    np.testing.assert_allclose([e.flow for e in flow.edges],reference["flow_rates"])
    np.testing.assert_allclose([e.permeability for e in flow.edges],reference["flow_permeability"])
    np.testing.assert_allclose([e.force for e in flow.edges],reference["flow_force"])
    left=np.array([[0,0,0],[0,.4,0],[0,.4,1],[0,0,1]],float); right=left.copy(); right[:,0]=1
    middle=np.array([[0,.1,.5],[1,.1,.5],[1,.8,.5],[0,.8,.5]],float)
    pipe3=pipe([left,right],[middle],method="center")
    np.testing.assert_allclose(pipe3.pipes,reference["pipe3_lines"])
    np.testing.assert_array_equal(pipe3.types,reference["pipe3_types"])
    np.testing.assert_allclose(pipe3.depths,reference["pipe3_depths"])
    np.testing.assert_array_equal(pipe3.intersection_pairs+1,reference["pipe3_pairs"])
    np.testing.assert_allclose(pipe3.intersections,reference["pipe3_intersections"])
    np.testing.assert_array_equal(pipe3.labels,reference["pipe3_labels"])
    np.testing.assert_array_equal(net([1,2,3],[4,5]),reference["net2"])
    np.testing.assert_array_equal(net([1,2],[3,4],[5,6]),reference["net3"])
    np.testing.assert_array_equal(stack([np.array([[1,2]]),np.array([[3,4]])]),reference["stacked"])
    np.testing.assert_array_equal(reshape(np.arange(1,17),4,4),reference["reshaped"])
    np.testing.assert_array_equal(flatri(reshape(np.arange(1,17),4,4)),reference["flattened_triangle"])
    np.testing.assert_array_equal(filter_ids([[1,2],[2,3],[2,3]]),reference["filtered_ids"])
    np.testing.assert_array_equal(not_indices([1,0,4,3,2,1],[0,2]),reference["not_values"])
    nx, ny = naned([[0, 1, 2, 3]])
    np.testing.assert_allclose(nx.reshape(-1), np.asarray(reference["naned_x"]).reshape(-1), equal_nan=True)
    np.testing.assert_allclose(ny.reshape(-1), np.asarray(reference["naned_y"]).reshape(-1), equal_nan=True)
    from pymrst.modules.adfne15 import distance, label, ray, relabel, rotate, translate, wrap
    points = np.array([[0., 0.], [1., 0.], [1., 1.]])
    np.testing.assert_allclose(translate(points, [2, 3]), reference["translated"])
    np.testing.assert_allclose(rotate(points, angle=90, center=[0, 0]), reference["rotated"], atol=1e-14)
    np.testing.assert_allclose(ray(points), reference["rays"])
    np.testing.assert_allclose(distance([[.5, .5], [2, .5]], [points]), reference["distances"])
    np.testing.assert_array_equal(label([[0, 2], [3]], 5), np.asarray(reference["labels"]).reshape(-1))
    np.testing.assert_array_equal(relabel([1, 1, -1, 2, 2, -2, 1, 2, 3]), np.asarray(reference["relabeled"]).reshape(-1))
    np.testing.assert_array_equal(wrap(np.arange(1, 11), 4), np.asarray(reference["wrapped"]).reshape(-1))
    from pymrst.modules.adfne15 import ellipse, get_shape_poly, inpoly, polyclip, sphere
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    np.testing.assert_array_equal(inpoly([[.5, .5], [1, .5], [2, .5]], square),
                                  np.asarray(reference["inpoly2"]).reshape(-1))
    square3 = np.column_stack((square, np.zeros(4)))
    np.testing.assert_array_equal(inpoly([[.5, .5, 0], [.5, .5, .1]], square3, 1e-8),
                                  np.asarray(reference["inpoly3"]).reshape(-1))
    np.testing.assert_allclose(ellipse(center=[1, 2], radii=[2, 1], resolution=4),
                               reference["ellipse4"], atol=1e-14)
    globe = sphere(center=[1, 2, 3], radius=2, resolution=6)
    sphere_values = reference["sphere6"].item()
    np.testing.assert_allclose(globe.x, sphere_values[0], atol=1e-14)
    np.testing.assert_allclose(globe.y, sphere_values[1], atol=1e-14)
    np.testing.assert_allclose(globe.z, sphere_values[2], atol=1e-14)
    cylinder = get_shape_poly(radius=1, center=[0, 0, 0], length=1, resolution=4)
    np.testing.assert_allclose(cylinder[0], reference["cylinder_cap1"])
    np.testing.assert_allclose(cylinder[2], reference["cylinder_side1"])
    candidate = np.array([[.5, -2, -2], [.5, 2, -2], [.5, 2, 2], [.5, -2, 2]])
    clipped, mask = polyclip([candidate], cylinder, return_mask=True)
    np.testing.assert_allclose(clipped[0], reference["clipped_domain_polygon"],
                               atol=1e-14)
    np.testing.assert_array_equal(mask, np.asarray(reference["clipped_domain_mask"]).reshape(-1))
    from pymrst.modules.adfne15 import CData, clock, convert, normalized, option, polygon_triangles, to_2d, to_3d
    assert clock(1000) == reference["convert_clock"]
    np.testing.assert_allclose(convert("deg", np.pi), reference["convert_deg"])
    np.testing.assert_allclose(convert("rad", 180), reference["convert_rad"])
    np.testing.assert_allclose(normalized([3, 4]), reference["convert_norm"])
    np.testing.assert_allclose(polygon_triangles(square)[0][0], reference["convert_tri1"])
    np.testing.assert_allclose(to_3d([[1, 2], [3, 4]]), reference["convert_3d"])
    points3 = np.column_stack((square, np.zeros(4)))
    np.testing.assert_allclose(to_2d(points3), reference["convert_2d"])
    opts = option(("fc", "red", "r", 2), "fc", "blue", "lw", 1)
    assert opts["radius"] == reference["option_radius"]
    assert opts["linewidth"] == reference["option_linewidth"]
    np.testing.assert_array_equal(CData.almost([0, 1], 0, .1), reference["cdata_almost"])
    columns = CData.deal([[1, 2], [3, 4]])
    np.testing.assert_array_equal(columns[0], reference["cdata_x"])
    np.testing.assert_array_equal(columns[1], reference["cdata_y"])
    np.testing.assert_array_equal(CData.get([[1, 2], [3, 4]], 10), reference["cdata_get"])
    leg = CData.leg([.4, .5, .6], .5, .01)
    for actual, name in zip(leg, ("cdata_less", "cdata_equal", "cdata_greater")):
        np.testing.assert_array_equal(actual, reference[name])
    np.testing.assert_allclose(CData.map([1, 2, 3], np.max, np.min, np.mean),
                               reference["cdata_map"])
    np.testing.assert_allclose(CData.array([0, 0, 1, 0], [[1, 2], [3, 4]])[0],
                               reference["cdata_array1"])
    from pymrst.modules.adfne15 import info, markov, stats, upscaling, variocloud, variogram, variomodel
    geo_info = info(np.array([[0., 0., 1., 0.], [0., 0., 0., 2.]]))
    matlab_info = reference["geo_info"].item()
    assert geo_info.count == matlab_info[0]
    np.testing.assert_allclose(geo_info.centers, matlab_info[1])
    np.testing.assert_allclose(geo_info.lengths, matlab_info[2])
    np.testing.assert_allclose(geo_info.angles, matlab_info[3])
    summary = stats([1, 2, np.nan, 4])
    if bool(reference["stats_toolbox_available"]):
        np.testing.assert_allclose(
            [summary.minimum, summary.mean, summary.maximum,
             summary.standard_deviation, summary.count],
            [reference["stats_min"], reference["stats_mean"],
             reference["stats_max"], reference["stats_std"],
             reference["stats_count"]])
    geo_points = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    geo_values = np.array([1., 2., 3., 4.])
    cloud = variocloud(geo_points, geo_values)
    np.testing.assert_allclose(cloud.distances, reference["cloud_d"])
    np.testing.assert_allclose(cloud.semivariances, reference["cloud_g"])
    np.testing.assert_allclose(cloud.variance, reference["cloud_v"])
    experimental = variogram(cloud.distances, cloud.semivariances,
                             cloud.variance, .5)
    np.testing.assert_allclose(experimental.lags, reference["variogram_h"])
    np.testing.assert_allclose(experimental.values, reference["variogram_v"])
    np.testing.assert_array_equal(experimental.counts, reference["variogram_p"])
    lags = np.asarray(reference["model_lags"])
    models = [
        ({"name":"sph","nugget":0,"sill":2,"range":1}, "model_sph"),
        ({"name":"exp","nugget":.1,"sill":2,"range":3}, "model_exp"),
        ({"name":"lin","nugget":1,"slope":2}, "model_lin")]
    for model, name in models:
        np.testing.assert_allclose(variomodel(model, lags), reference[name])
    np.testing.assert_array_equal(markov([[1, 0], [0, 1]], 1, 10,
                                         rng=np.random.default_rng(1)),
                                  reference["markov_fixed"])
    values = np.arange(1, 17, dtype=float).reshape(4, 4)
    np.testing.assert_allclose(upscaling(values, "arithmetic", 2),
                               reference["upscale_arithmetic"])
    from pymrst.modules.adfne15 import CExt, grid
    np.testing.assert_allclose(grid(4), reference["grid4"])
    definition = np.array([0,0,0,1,0,0,0,1,0], float)
    np.testing.assert_allclose(CExt.plane_normal(definition), reference["cext_normal"])
    np.testing.assert_allclose(CExt.intersect_line_plane([[0,0,1,0,0,-2]], definition),
                               np.asarray(reference["cext_line_plane"]).reshape(1,3))
    np.testing.assert_allclose(CExt.point_on_plane([[1,2,3]], definition),
                               np.asarray(reference["cext_projected"]).reshape(1,3))
    assert bool(reference["mesh_unavailable"])
    with pytest.raises(ADFNEMeshUnavailableError): mesh([[0,0,1,1]],.1)
