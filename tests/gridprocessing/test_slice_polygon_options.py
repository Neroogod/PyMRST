import numpy as np

from pymrst.gridprocessing import cart_grid, compute_grid_slice_polygons


def test_piecewise_polyline_reports_each_segment():
    grid = cart_grid([3, 3, 1], [3.0, 3.0, 1.0])
    polygons = compute_grid_slice_polygons(
        grid,
        np.array([[0.0, 0.5, 0.0], [2.5, 0.5, 0.0], [2.5, 3.0, 0.0]]),
        cut_dir=np.array([0.0, 0.0, 1.0]),
    )
    assert set(polygons["segment_ix"]) == {0, 1}
    assert polygons["cell_ix"].size == polygons["segment_ix"].size
    assert np.all(np.diff(polygons["node_pos"]) >= 3)


def test_face_and_cell_subsets_restrict_output():
    grid = cart_grid([3, 1, 1], [3.0, 1.0, 1.0])
    all_polygons = compute_grid_slice_polygons(
        grid, [1.5, 0.5, 0.5], normal=[1.0, 0.0, 0.0],
    )
    selected = compute_grid_slice_polygons(
        grid, [1.5, 0.5, 0.5], normal=[1.0, 0.0, 0.0], cell_ix=[1],
    )
    assert all_polygons["cell_ix"].size == 1
    assert np.array_equal(selected["cell_ix"], [1])


def test_segment_radius_clips_polygon_width():
    grid = cart_grid([1, 1, 1], [1.0, 1.0, 1.0])
    polygons = compute_grid_slice_polygons(
        grid,
        np.array([[0.5, 0.0, 0.5], [0.5, 1.0, 0.5]]),
        cut_dir=np.array([0.0, 0.0, 1.0]),
        radius=0.2,
    )
    assert np.max(np.abs(polygons["coords_2d"][:, 1])) <= 0.2 + 1e-14
    assert np.isclose(polygons["coords_3d"][:, 2].min(), 0.3)
    assert np.isclose(polygons["coords_3d"][:, 2].max(), 0.7)
