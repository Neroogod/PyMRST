import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing import process_grdecl
from pymrst.gridprocessing.testgrids import (
    extruded_triangle_grid,
    one_sloping_fault,
)
from pymrst.modules.upscaling.coarsegrid import (
    generate_coarse_grid,
    partition_cart_grid,
)
from pymrst.plotting.private.plot_face_outline import face_outline_geometry
from pymrst.plotting.private.plot_patches import (
    build_patch_geometry,
    normalize_color_data,
)
from pymrst.plotting.plot_faces import plot_faces
from pymrst.plotting.plot_face_data import plot_face_data
from pymrst.plotting.plot_cell_data import plot_cell_data
from pymrst.plotting.plot_grid import plot_grid
from pymrst.plotting.plot_node_data import plot_node_data
from pymrst.plotting.plot_well import plot_well
from pymrst.plotting.plot_faults import plot_faults
from pymrst.plotting.outline_coarse_grid import (
    coarse_partition_faces,
    outline_coarse_grid,
)
from pymrst.plotting.plot_block_and_neighbors import (
    coarse_block_neighbors,
    plot_block_and_neighbors,
)
from pymrst.plotting.plot_slice import plot_slice, slice_entities
from pymrst.plotting.interpolate_data import interpolate_data
from pymrst.plotting.unstructured_surf import (
    surface_geometry,
    unstructured_surf,
)
from pymrst.plotting.unstructured_contour import (
    contour_grid,
    unstructured_contour,
)
from pymrst.plotting.plot_contours import contour_face_sets, plot_contours
from pymrst.plotting.colorbar_hist import (
    colorbar_hist,
    histogram_colorbar_data,
)
from pymrst.plotting.mrst_colorbar import mrst_colorbar
from pymrst.plotting.plot_grid_volumes import (
    _histogram_centers,
    plot_grid_volumes,
    volume_samples,
)
from pymrst.params.wells_and_bc import Well
from pymrst.gridprocessing import process_faults
from pymrst.plotting.boundary_faces import boundary_faces
from pymrst.plotting.private.rendering_compat import (
    configure_reservoir_camera,
    translate_matlab_properties,
)


def test_patch_geometry_compacts_selected_3d_face_nodes():
    grid = compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
    face = np.array([0])
    geometry = build_patch_geometry(grid, face)

    assert geometry.entity_ids.tolist() == [0]
    assert geometry.vertices.shape == (4, 3)
    assert len(geometry.connectivity) == 1
    np.testing.assert_array_equal(
        np.sort(geometry.connectivity[0]), np.arange(4)
    )
    np.testing.assert_array_equal(
        geometry.vertices, grid.nodes.coords[geometry.node_ids]
    )


def test_matlab_render_properties_translate_case_insensitively():
    translated = translate_matlab_properties({
        "FaceColor": "red",
        "EdgeColor": "none",
        "FaceAlpha": 0.35,
        "LineWidth": 4,
    })
    assert translated == {
        "color": "red",
        "edge_color": None,
        "show_edges": False,
        "opacity": 0.35,
        "line_width": 4,
    }
    assert translate_matlab_properties({
        "FaceAlpha": 0.2, "opacity": 0.8,
    })["opacity"] == 0.8


def test_plot_grid_applies_zscale_matlab_properties_and_depth_down_camera():
    class Camera:
        up = (0.0, 1.0, 0.0)
        azimuth = 0
        elevation = 0

    class Plotter:
        def __init__(self):
            self.camera = Camera()
            self.camera_position = None
            self.kwargs = None

        def reset_camera(self):
            self.reset = True

        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh
            self.kwargs = kwargs
            return "actor"

    grid = compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 2.0]))
    plotter = Plotter()
    result = plot_grid(
        grid, plotter=plotter, show=False, zscale=3.0,
        **{"FaceColor": "cyan", "EdgeColor": "none",
           "FaceAlpha": 0.4, "LineWidth": 3},
    )
    assert np.ptp(result.geometry.vertices[:, 2]) == pytest.approx(6.0)
    assert np.ptp(plotter.mesh.points[:, 2]) == pytest.approx(6.0)
    assert plotter.kwargs["color"] == (0.0, 1.0, 1.0)
    assert plotter.kwargs["opacity"] == pytest.approx(0.4)
    assert plotter.kwargs["line_width"] == pytest.approx(3.0)
    assert plotter.kwargs["show_edges"] is False
    # Supplied plotters retain their position, but always get depth-down up.
    assert plotter.camera.up == (0.0, 0.0, -1.0)


def test_canonical_camera_sets_reproducible_reversed_z_view():
    class Camera:
        up = None
        azimuth = 0
        elevation = 0

    class Plotter:
        camera = Camera()
        camera_position = None

        def reset_camera(self):
            self.reset = True

    plotter = Plotter()
    configure_reservoir_camera(plotter, flip_z=True, reset=True)
    assert plotter.camera_position == "xz"
    assert plotter.camera.up == (0.0, 0.0, -1.0)
    assert plotter.camera.azimuth == 210
    assert plotter.camera.elevation == 25


def test_logical_face_mask_and_empty_selection():
    grid = compute_geometry(cart_grid([2, 1], [2.0, 1.0]))
    mask = np.zeros(grid.faces.num, dtype=bool)
    mask[[0, 2]] = True
    geometry = build_patch_geometry(grid, mask)
    assert geometry.entity_ids.tolist() == [0, 2]

    empty = build_patch_geometry(grid, [])
    assert empty.n_entities == 0
    assert empty.connectivity == ()
    assert empty.vertices.shape == (0, 2)


def test_invalid_face_selections_are_rejected():
    grid = compute_geometry(cart_grid([2, 1], [2.0, 1.0]))
    with pytest.raises(ValueError, match="logical face mask"):
        build_patch_geometry(grid, np.ones(grid.faces.num-1, dtype=bool))
    with pytest.raises(IndexError, match="face indices"):
        build_patch_geometry(grid, [-1])
    with pytest.raises(ValueError, match="integers"):
        build_patch_geometry(grid, [0.5])


def test_face_and_node_color_modes_are_unambiguous():
    grid = compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
    geometry = build_patch_geometry(grid, [0, 1])

    face = normalize_color_data(grid, geometry, np.array([2.0, 3.0]))
    assert face.mode == "face"
    np.testing.assert_array_equal(face.values[:, 0], [2.0, 3.0])

    rgb = normalize_color_data(
        grid, geometry, np.tile([1.0, 0.0, 0.0], (2, 1))
    )
    assert rgb.mode == "face" and rgb.values.shape == (2, 3)

    nodal_values = np.arange(grid.nodes.num, dtype=float)
    nodal = normalize_color_data(grid, geometry, nodal_values)
    assert nodal.mode == "node"
    np.testing.assert_array_equal(
        nodal.values[:, 0], nodal_values[geometry.node_ids]
    )
    scalar = normalize_color_data(grid, geometry, 4.0)
    assert scalar.mode == "face"
    np.testing.assert_array_equal(scalar.values[:, 0], [4.0, 4.0])


def test_outline_removes_edge_shared_by_two_selected_faces():
    grid = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
    centers = np.asarray(grid.faces.centroids)
    boundary = np.any(np.asarray(grid.faces.neighbors) < 0, axis=1)
    bottom = np.flatnonzero(boundary & np.isclose(centers[:, 2], 0.0))
    assert bottom.size == 2

    outline = face_outline_geometry(grid, bottom)
    assert len(outline.connectivity) == 6
    undirected = {
        tuple(sorted(map(int, edge))) for edge in outline.connectivity
    }
    assert len(undirected) == 6


def test_plot_faces_builds_vtk_mesh_with_injected_plotter():
    class Plotter:
        def __init__(self):
            self.mesh = None
            self.kwargs = None

        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh
            self.kwargs = kwargs
            return "actor"

    grid = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
    faces = boundary_faces(grid)
    plotter = Plotter()
    result = plot_faces(
        grid,
        color=np.arange(faces.size, dtype=float)[:, None],
        plotter=plotter,
        show=False,
    )
    assert result.geometry.n_entities == faces.size == 10
    assert result.actor == "actor"
    assert plotter.mesh.n_cells == faces.size
    np.testing.assert_array_equal(
        plotter.mesh.cell_data["face_data"], np.arange(faces.size)
    )
    assert plotter.kwargs["scalars"] == "face_data"


def test_2d_faces_are_vtk_lines_not_vertex_cells():
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh
            return "lines"

    grid = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
    plotter = Plotter()
    result = plot_faces(grid, plotter=plotter, show=False)

    assert result.actor == "lines"
    assert result.geometry.n_entities == grid.faces.num
    assert plotter.mesh.n_lines == grid.faces.num
    assert plotter.mesh.n_verts == 0


def test_facecolor_none_is_wireframe_and_outline_is_collection_boundary():
    class Plotter:
        def __init__(self):
            self.calls = []

        def add_mesh(self, mesh, **kwargs):
            self.calls.append((mesh, kwargs))
            return len(self.calls)

    grid = compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
    plotter = Plotter()
    result = plot_faces(
        grid,
        face_color="none",
        outline=True,
        plotter=plotter,
        show=False,
    )

    assert result.actor == 1
    assert plotter.calls[0][1]["style"] == "wireframe"
    # The six faces form a closed shell, so its collection boundary is empty.
    assert result.outline_actor is None


def test_coarse_faces_expand_to_grouped_fine_faces_and_colors():
    class Plotter:
        def __init__(self):
            self.calls = []

        def add_mesh(self, mesh, **kwargs):
            self.calls.append((mesh, kwargs))
            return len(self.calls)

    fine = compute_geometry(cart_grid([4, 2, 1], [4.0, 2.0, 1.0]))
    partition = partition_cart_grid([4, 2, 1], [2, 1, 1])
    coarse = generate_coarse_grid(fine, partition)
    coarse_boundary = boundary_faces(coarse.g)
    colors = np.arange(coarse_boundary.size, dtype=float)[:, None]
    plotter = Plotter()

    result = plot_faces(
        coarse, color=colors, plotter=plotter, show=False
    )

    fine_boundary = boundary_faces(fine)
    np.testing.assert_array_equal(
        np.sort(result.geometry.entity_ids), fine_boundary
    )
    assert result.geometry.group_ids.size == fine_boundary.size
    np.testing.assert_array_equal(
        plotter.calls[0][0].cell_data["face_data"],
        colors[result.geometry.group_ids, 0],
    )
    assert plotter.calls[0][1]["show_edges"] is False
    assert result.outline_actor == 2


def test_internal_coarse_face_maps_only_to_interface_fine_faces():
    fine = compute_geometry(cart_grid([4, 2, 1], [4.0, 2.0, 1.0]))
    coarse = generate_coarse_grid(
        fine, partition_cart_grid([4, 2, 1], [2, 1, 1])
    )
    internal = np.flatnonzero(
        np.all(np.asarray(coarse.g.faces.neighbors) >= 0, axis=1)
    )
    geometry = plot_faces(
        coarse,
        internal,
        plotter=type("P", (), {"add_mesh": lambda self, mesh, **kw: mesh})(),
        show=False,
    ).geometry
    assert internal.size == 1
    assert geometry.n_entities == 2
    assert np.all(
        np.all(np.asarray(fine.faces.neighbors)[geometry.entity_ids] >= 0,
               axis=1)
    )


def test_faulted_corner_point_and_polyhedral_fixtures_keep_face_loops():
    faulted = compute_geometry(process_grdecl(one_sloping_fault([4, 2, 2], 2)))
    fault_geometry = build_patch_geometry(
        faulted, boundary_faces(faulted)
    )
    assert fault_geometry.n_entities > 0
    assert all(loop.size >= 3 for loop in fault_geometry.connectivity)
    assert fault_geometry.node_ids.size < faulted.nodes.num

    polyhedral = extruded_triangle_grid(10.0, seed=2)
    poly_geometry = build_patch_geometry(
        polyhedral, boundary_faces(polyhedral)
    )
    sizes = {loop.size for loop in poly_geometry.connectivity}
    assert 3 in sizes
    assert all(size >= 3 for size in sizes)


def test_plot_face_data_maps_full_face_values_to_subset_boundary():
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh
            self.kwargs = kwargs
            return "face-data"

    grid = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
    data = np.arange(grid.faces.num, dtype=float)
    plotter = Plotter()
    result = plot_face_data(
        grid, [1], data, plotter=plotter, show=False
    )
    expected = boundary_faces(grid, [1])

    np.testing.assert_array_equal(result.geometry.entity_ids, expected)
    np.testing.assert_array_equal(
        plotter.mesh.cell_data["face_data"], data[expected]
    )
    assert result.actor == "face-data"
    assert plotter.kwargs["show_edges"] is False


def test_plot_face_data_accepts_logical_cells_rgb_and_empty_selection():
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh
            self.kwargs = kwargs
            return "rgb"

    grid = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
    mask = np.array([True, False])
    rgb = np.tile([0.2, 0.4, 0.8], (grid.faces.num, 1))
    plotter = Plotter()
    result = plot_face_data(
        grid, mask, rgb, plotter=plotter, show=False
    )
    assert result.geometry.n_entities == boundary_faces(grid, mask).size
    assert plotter.kwargs["rgb"] is True

    empty = plot_face_data(
        grid, [], np.arange(grid.faces.num), plotter=plotter, show=False
    )
    assert empty.geometry.n_entities == 0
    assert empty.actor is None


def test_plot_face_data_validates_dimension_and_full_face_data_shape():
    grid3 = compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
    with pytest.raises(ValueError, match="expected"):
        plot_face_data(
            grid3, np.ones(grid3.faces.num-1),
            plotter=object(), show=False,
        )

    grid2 = compute_geometry(cart_grid([1, 1], [1.0, 1.0]))
    with pytest.raises(ValueError, match="3-D"):
        plot_face_data(
            grid2, np.ones(grid2.faces.num),
            plotter=object(), show=False,
        )


def test_plot_cell_data_3d_maps_subset_values_to_boundary_owners():
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh
            self.kwargs = kwargs
            return "cells-3d"

    grid = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
    selected = np.array([1, 2])
    subset_data = np.array([10.0, 20.0])
    plotter = Plotter()
    result = plot_cell_data(
        grid, subset_data, selected, plotter=plotter, show=False
    )
    faces, owners = boundary_faces(grid, selected, with_cells=True)
    expected = np.where(owners == 1, 10.0, 20.0)

    np.testing.assert_array_equal(result.geometry.entity_ids, faces)
    np.testing.assert_array_equal(
        plotter.mesh.cell_data["face_data"], expected
    )
    assert result.actor == "cells-3d"


def test_plot_cell_data_excludes_nan_cells_before_boundary_extraction():
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh
            return "finite"

    grid = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
    data = np.array([1.0, np.nan, 3.0])
    plotter = Plotter()
    result = plot_cell_data(
        grid, data, plotter=plotter, show=False
    )
    expected = boundary_faces(grid, [0, 2])
    np.testing.assert_array_equal(result.geometry.entity_ids, expected)
    assert not np.isnan(plotter.mesh.cell_data["face_data"]).any()


def test_plot_cell_data_2d_builds_cell_polygons_and_rgb():
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh
            self.kwargs = kwargs
            return "cells-2d"

    grid = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
    selected = np.array([0, 3])
    rgb = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    plotter = Plotter()
    result = plot_cell_data(
        grid, rgb, selected, plotter=plotter, show=False
    )

    np.testing.assert_array_equal(result.geometry.entity_ids, selected)
    assert all(loop.size == 4 for loop in result.geometry.connectivity)
    assert plotter.mesh.n_cells == 2
    np.testing.assert_array_equal(plotter.mesh.cell_data["cell_data"], rgb)
    assert plotter.kwargs["rgb"] is True


def test_plot_cell_data_validates_masks_indices_and_data_rows():
    grid = compute_geometry(cart_grid([2, 1], [2.0, 1.0]))
    with pytest.raises(ValueError, match="logical cell mask"):
        plot_cell_data(grid, [1.0], [True], plotter=object(), show=False)
    with pytest.raises(IndexError, match="cell indices"):
        plot_cell_data(grid, [1.0], [-1], plotter=object(), show=False)
    with pytest.raises(ValueError, match="cell data rows"):
        plot_cell_data(
            grid, np.ones(3), [0], plotter=object(), show=False
        )


def test_plot_grid_3d_uses_only_subset_boundary_faces():
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh
            self.kwargs = kwargs
            return "grid-3d"

    grid = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
    mask = np.array([False, True, True])
    plotter = Plotter()
    result = plot_grid(grid, mask, plotter=plotter, show=False)

    expected = boundary_faces(grid, mask)
    np.testing.assert_array_equal(result.geometry.entity_ids, expected)
    assert plotter.mesh.n_cells == expected.size
    assert result.actor == "grid-3d"


def test_plot_grid_2d_uses_selected_cell_polygons():
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh
            self.kwargs = kwargs
            return "grid-2d"

    grid = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
    selected = np.array([1, 2])
    plotter = Plotter()
    result = plot_grid(
        grid,
        selected,
        color="cyan",
        show_edges=False,
        plotter=plotter,
        show=False,
    )
    np.testing.assert_array_equal(result.geometry.entity_ids, selected)
    assert plotter.mesh.n_cells == selected.size
    np.testing.assert_allclose(
        plotter.mesh.cell_data["cell_data"],
        np.tile([0.0, 1.0, 1.0], (2, 1)),
    )
    assert plotter.kwargs["show_edges"] is False


def test_plot_node_data_3d_compacts_selected_boundary_nodes():
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh
            self.kwargs = kwargs
            return "nodes-3d"

    grid = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
    values = np.arange(grid.nodes.num, dtype=float)
    plotter = Plotter()
    result = plot_node_data(
        grid, values, [0], plotter=plotter, show=False
    )
    expected_faces = boundary_faces(grid, [0])
    np.testing.assert_array_equal(result.geometry.entity_ids, expected_faces)
    np.testing.assert_array_equal(
        plotter.mesh.point_data["node_data"],
        values[result.geometry.node_ids],
    )
    assert plotter.kwargs["preference"] == "point"


def test_plot_node_data_2d_rgb_and_displacement():
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh
            self.kwargs = kwargs
            return "nodes-2d"

    grid = compute_geometry(cart_grid([2, 1], [2.0, 1.0]))
    rgb = np.column_stack([
        np.linspace(0.0, 1.0, grid.nodes.num),
        np.zeros(grid.nodes.num),
        np.ones(grid.nodes.num),
    ])
    displacement = np.tile([0.25, -0.5], (grid.nodes.num, 1))
    plotter = Plotter()
    result = plot_node_data(
        grid,
        rgb,
        [1],
        uu=displacement,
        plotter=plotter,
        show=False,
    )
    np.testing.assert_allclose(
        result.geometry.vertices[:, :2],
        grid.nodes.coords[result.geometry.node_ids] +
        displacement[result.geometry.node_ids],
    )
    np.testing.assert_array_equal(
        plotter.mesh.point_data["node_data"],
        rgb[result.geometry.node_ids],
    )
    assert plotter.kwargs["rgb"] is True
    # Plotting displacement must never mutate simulation geometry.
    assert not np.allclose(
        grid.nodes.coords,
        grid.nodes.coords + displacement,
    )


def test_plot_node_data_validates_data_and_displacement_shapes():
    grid = compute_geometry(cart_grid([1, 1], [1.0, 1.0]))
    with pytest.raises(ValueError, match="node data"):
        plot_node_data(
            grid, np.ones(grid.nodes.num-1), plotter=object(), show=False
        )
    with pytest.raises(ValueError, match="uu has shape"):
        plot_node_data(
            grid,
            np.ones(grid.nodes.num),
            uu=np.zeros((grid.nodes.num, 3)),
            plotter=object(),
            show=False,
        )


def test_plot_cell_data_1d_matches_centroid_value_line():
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh
            self.kwargs = kwargs
            return "line-data"

    grid = compute_geometry(cart_grid([4], [8.0]))
    selected = np.array([1, 2, 3])
    values = np.array([5.0, 7.0, 11.0])
    plotter = Plotter()
    result = plot_cell_data(
        grid, values, selected, plotter=plotter, show=False
    )
    np.testing.assert_allclose(
        result.geometry.vertices,
        np.column_stack([grid.cells.centroids[selected, 0], values]),
    )
    np.testing.assert_array_equal(result.geometry.entity_ids, selected)
    assert plotter.mesh.n_lines == 1
    assert result.actor == "line-data"


def test_plot_grid_1d_uses_selected_centroids_at_unit_height():
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh
            self.kwargs = kwargs
            return "line-grid"

    grid = compute_geometry(cart_grid([4], [8.0]))
    mask = np.array([True, False, True, True])
    plotter = Plotter()
    result = plot_grid(
        grid,
        mask,
        color="magenta",
        line_width=3.0,
        plotter=plotter,
        show=False,
    )
    selected = np.flatnonzero(mask)
    np.testing.assert_allclose(
        result.geometry.vertices[:, 0],
        grid.cells.centroids[selected, 0],
    )
    np.testing.assert_array_equal(result.geometry.vertices[:, 1], 1.0)
    assert plotter.kwargs["color"] == "magenta"
    assert plotter.kwargs["line_width"] == 3.0


def _well(cells, *, name, sign, direction="z"):
    cells = np.asarray(cells, dtype=np.int64)
    return Well(
        cells=cells,
        type="rate",
        val=1.0,
        r=np.full(cells.size, 0.1),
        dir=direction,
        WI=np.ones(cells.size),
        dZ=np.zeros(cells.size),
        name=name,
        compi=np.array([1.0, 0.0]),
        refDepth=0.0,
        sign=sign,
    )


def test_plot_well_geometry_colors_labels_and_handle_groups():
    class Plotter:
        def __init__(self):
            self.mesh_calls = []
            self.label_calls = []

        def add_mesh(self, mesh, **kwargs):
            self.mesh_calls.append((mesh, kwargs))
            return f"mesh-{len(self.mesh_calls)}"

        def add_point_labels(self, points, labels, **kwargs):
            self.label_calls.append((points, labels, kwargs))
            return f"label-{len(self.label_calls)}"

    grid = compute_geometry(cart_grid([1, 1, 3], [1.0, 1.0, 3.0]))
    injector = _well([0, 1, 2], name="INJ_1", sign=1)
    producer = _well([2], name="PROD_1", sign=-1)
    plotter = Plotter()
    result = plot_well(
        grid,
        [injector, producer],
        plotter,
        height=2.0,
        color="blue",
        color2="green",
        fontsize=12,
    )

    assert len(result.geometry) == 2
    assert result.geometry[0].color == "blue"
    assert result.geometry[1].color == "green"
    assert result.geometry[0].name == "INJ_1"
    assert result.geometry[0].head[2] == pytest.approx(-2.0)
    np.testing.assert_array_equal(
        result.geometry[0].completion_points,
        grid.cells.centroids[[0, 1, 2]],
    )
    assert len(result.head_actors) == 2
    assert len(result.text_actors) == 2
    assert len(result.path_actors) == 2
    assert len(result.line_actors) == 2
    assert len(plotter.label_calls) == 2


def test_plot_well_validates_dimension_cells_and_cylinder_resolution():
    grid2 = compute_geometry(cart_grid([1, 1], [1.0, 1.0]))
    well = _well([0], name="W1", sign=1)
    with pytest.raises(ValueError, match="3-D"):
        plot_well(grid2, well, object())

    grid3 = compute_geometry(cart_grid([1, 1, 1]))
    with pytest.raises(IndexError, match="invalid cell"):
        plot_well(
            grid3,
            _well([2], name="bad", sign=1),
            object(),
        )
    with pytest.raises(ValueError, match="cylpts"):
        plot_well(grid3, well, object(), cylpts=2)


@pytest.mark.parametrize(
    ("dims", "extent", "direction", "constant_axis", "radial_axes"),
    [
        ([2, 1, 1], [2.0, 1.0, 1.0], "x", 0, (1, 2)),
        ([1, 2, 1], [1.0, 2.0, 1.0], "y", 1, (0, 2)),
        ([1, 1, 2], [1.0, 1.0, 2.0], "z", 2, (0, 1)),
    ],
)
def test_plot_well_builds_direction_dependent_mrst_cylinders(
    dims, extent, direction, constant_axis, radial_axes
):
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            return mesh

        def add_point_labels(self, *args, **kwargs):
            return None

    grid = compute_geometry(cart_grid(dims, extent))
    result = plot_well(
        grid,
        _well([0, 1], name=direction.upper(), sign=1,
              direction=direction),
        Plotter(),
        cylpts=8,
        fontsize=0,
    )
    geometry = result.geometry[0]
    assert geometry.directions == (direction, direction)
    assert geometry.surface_faces.shape == (8, 4)
    rows = geometry.surface_vertices.reshape((2, 9, 3), order="F")
    np.testing.assert_allclose(
        rows[:, :, constant_axis],
        np.repeat(
            grid.cells.centroids[[0, 1], constant_axis, None], 9, axis=1
        ),
    )
    for axis in radial_axes:
        assert np.ptp(rows[0, :, axis]) > 0.0


def test_plot_well_builds_mrst_slanted_segment_branch():
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            return mesh

        def add_point_labels(self, *args, **kwargs):
            return None

    grid = compute_geometry(cart_grid([2, 1, 2], [2.0, 1.0, 2.0]))
    geometry = plot_well(
        grid,
        _well([0, 3], name="SLANT", sign=1, direction="qq"),
        Plotter(), cylpts=8, fontsize=0,
    ).geometry[0]
    rows = geometry.surface_vertices.reshape((2, 9, 3), order="F")
    assert geometry.directions == ("q", "q")
    np.testing.assert_allclose(
        rows[0, :, 2],
        grid.cells.centroids[0, 2] - 0.5*geometry.vertical_radius,
    )
    np.testing.assert_allclose(
        rows[1, :, 2],
        grid.cells.centroids[3, 2] + 0.5*geometry.vertical_radius,
    )


def test_plot_well_single_completion_uses_anisotropic_sphere_and_disk_caps():
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            return mesh

        def add_point_labels(self, *args, **kwargs):
            return None

    grid = compute_geometry(cart_grid([1, 1, 1], [2.0, 3.0, 4.0]))
    geometry = plot_well(
        grid, _well([0], name="ONE", sign=1), Plotter(),
        cylpts=10, fontsize=0,
    ).geometry[0]
    assert geometry.directions == ("s",)
    assert geometry.surface_vertices.shape == (121, 3)
    assert geometry.surface_faces.shape == (100, 4)
    assert geometry.head_vertices.shape == (11, 3)
    assert geometry.heel_vertices.shape == (11, 3)
    assert np.ptp(geometry.surface_vertices[:, 2]) == pytest.approx(
        2.0*geometry.vertical_radius
    )


def test_plot_well_skips_closed_wells_and_filters_closed_or_repeated_completions():
    class Plotter:
        def __init__(self):
            self.calls = 0

        def add_mesh(self, mesh, **kwargs):
            self.calls += 1
            return mesh

        def add_point_labels(self, *args, **kwargs):
            return None

    grid = compute_geometry(cart_grid([1, 1, 2]))
    closed = _well([0, 1], name="CLOSED", sign=1)
    closed.status = False
    closed_plotter = Plotter()
    result = plot_well(grid, closed, closed_plotter, fontsize=0)
    assert not result.geometry[0].active
    assert closed_plotter.calls == 0
    assert result.path_actors == (None,)

    repeated = _well([0, 0, 1], name="REPEAT", sign=1, direction="xyz")
    repeated.cstatus = np.array([True, True, True])
    geometry = plot_well(
        grid, repeated, Plotter(), cylpts=6, fontsize=0,
    ).geometry[0]
    assert geometry.completion_points.shape[0] == 3
    assert geometry.trajectory.shape[0] == 3  # head plus two unique cells
    assert geometry.surface_faces.shape == (6, 4)
    assert geometry.directions == ("x", "z")

    filtered = _well([0, 1], name="FILTERED", sign=1, direction="xy")
    filtered.cstatus = np.array([False, True])
    geometry = plot_well(
        grid, filtered, Plotter(), cylpts=6, fontsize=0,
    ).geometry[0]
    np.testing.assert_array_equal(geometry.cells, [1])
    assert geometry.directions == ("s",)


def test_plot_faults_groups_faces_and_places_named_labels():
    class Plotter:
        def __init__(self):
            self.labels = None

        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh
            self.kwargs = kwargs
            return "fault-faces"

        def add_point_labels(self, points, labels, **kwargs):
            self.labels = (points, labels, kwargs)
            return "fault-labels"

    grid = compute_geometry(cart_grid([3, 3, 2]))
    faults, _ = process_faults(grid, {
        "FAULTS": [
            ["BOUND", 1, 1, 1, 3, 1, 2, "X-"],
            ["MAIN", 2, 2, 1, 2, 1, 2, "X+"],
        ],
    })
    plotter = Plotter()
    result = plot_faults(
        grid,
        faults,
        height=2.5,
        color="orange",
        plotter=plotter,
        show=False,
    )

    expected_faces = np.concatenate([fault.faces for fault in faults])
    np.testing.assert_array_equal(result.faces.geometry.entity_ids,
                                  expected_faces)
    assert result.names == tuple(fault.name for fault in faults)
    assert result.face_groups.size == expected_faces.size
    expected_labels = np.asarray([
        grid.faces.centroids[fault.faces].mean(axis=0) for fault in faults
    ])
    expected_labels[:, 2] -= 2.5
    np.testing.assert_allclose(result.label_positions, expected_labels)
    assert result.faces.actor == "fault-faces"
    assert result.text_actor == "fault-labels"
    assert plotter.labels[1] == list(result.names)


def test_plot_faults_accepts_mappings_and_validates_inputs():
    grid = compute_geometry(cart_grid([1, 1, 1]))
    face = int(boundary_faces(grid)[0])
    mapping = {"Name": "F", "Faces": [face]}

    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            return "actor"

    result = plot_faults(
        grid, [mapping], fontsize=0, plotter=Plotter(), show=False
    )
    assert result.names == ("F",)
    assert result.text_actor is None

    with pytest.raises(IndexError, match="invalid face"):
        plot_faults(
            grid,
            [{"name": "bad", "faces": [grid.faces.num]}],
            plotter=Plotter(),
            show=False,
        )


def test_coarse_partition_faces_2d_include_outer_boundary_and_interface():
    grid = compute_geometry(cart_grid([4, 2], [4.0, 2.0]))
    partition = partition_cart_grid([4, 2], [2, 1])
    selected = coarse_partition_faces(grid, partition)
    boundary = boundary_faces(grid)
    interior = selected[
        np.all(grid.faces.neighbors[selected] >= 0, axis=1)
    ]

    assert set(boundary).issubset(set(selected))
    assert interior.size == 2
    left = partition[grid.faces.neighbors[interior, 0]]
    right = partition[grid.faces.neighbors[interior, 1]]
    assert np.all(left != right)


def test_coarse_partition_faces_3d_exclude_outer_boundary():
    grid = compute_geometry(cart_grid([4, 2, 1], [4.0, 2.0, 1.0]))
    partition = partition_cart_grid([4, 2, 1], [2, 1, 1])
    selected = coarse_partition_faces(grid, partition)

    assert selected.size == 2
    assert np.all(np.all(grid.faces.neighbors[selected] >= 0, axis=1))
    assert not np.intersect1d(selected, boundary_faces(grid)).size


def test_outline_coarse_grid_renders_exact_selected_entities():
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh
            self.kwargs = kwargs
            return "coarse-outline"

    grid = compute_geometry(cart_grid([4, 2], [4.0, 2.0]))
    partition = partition_cart_grid([4, 2], [2, 1])
    plotter = Plotter()
    result = outline_coarse_grid(
        grid,
        partition,
        color="blue",
        plotter=plotter,
        show=False,
    )
    np.testing.assert_array_equal(
        result.geometry.entity_ids,
        coarse_partition_faces(grid, partition),
    )
    assert result.actor == "coarse-outline"
    assert plotter.mesh.n_lines == result.geometry.entity_ids.size


def test_coarse_partition_faces_validates_partition_shape_and_values():
    grid = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
    with pytest.raises(ValueError, match="expected"):
        coarse_partition_faces(grid, [0, 1])
    with pytest.raises(ValueError, match="finite"):
        coarse_partition_faces(grid, [0, 0, 1, np.nan])


def test_plot_block_and_neighbors_selects_face_adjacent_blocks():
    class Plotter:
        def __init__(self):
            self.calls = []

        def add_mesh(self, mesh, **kwargs):
            self.calls.append((mesh, kwargs))
            return len(self.calls)

    fine = compute_geometry(cart_grid([6, 2, 1], [6.0, 2.0, 1.0]))
    coarse = generate_coarse_grid(
        fine, partition_cart_grid([6, 2, 1], [3, 1, 1])
    )
    np.testing.assert_array_equal(coarse_block_neighbors(coarse.g, 1), [0, 2])
    plotter = Plotter()
    result = plot_block_and_neighbors(
        coarse,
        1,
        plot_faults=(False, False),
        alpha=(0.8, 0.5),
        plotter=plotter,
        show=False,
    )
    np.testing.assert_array_equal(result.neighbors, [0, 2])
    np.testing.assert_array_equal(
        result.block_faces, boundary_faces(fine, coarse.partition == 1)
    )
    assert len(result.neighbor_faces) == 2
    assert len(result.neighbor_actors) == 2
    assert result.fault_actors == ()
    assert len(plotter.calls) == 3


def test_plot_block_and_neighbors_extracts_tagged_fault_faces():
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            return "actor"

    fine = compute_geometry(cart_grid([4, 1, 1], [4.0, 1.0, 1.0]))
    coarse = generate_coarse_grid(
        fine, partition_cart_grid([4, 1, 1], [2, 1, 1])
    )
    fine.faces.tag = np.zeros(fine.faces.num, dtype=np.int64)
    candidate = boundary_faces(fine, coarse.partition == 0)[0]
    fine.faces.tag[candidate] = 1
    result = plot_block_and_neighbors(
        coarse,
        0,
        plot_faults=(True, False),
        alpha=(1.0, 1.0, 0.4),
        plotter=Plotter(),
        show=False,
    )
    np.testing.assert_array_equal(result.block_fault_faces, [candidate])
    assert candidate not in result.block_faces
    assert len(result.fault_actors) == 1


def test_plot_block_and_neighbors_validates_options_and_block():
    fine = compute_geometry(cart_grid([2, 1, 1]))
    coarse = generate_coarse_grid(
        fine, partition_cart_grid([2, 1, 1], [2, 1, 1])
    )
    with pytest.raises(IndexError, match="block"):
        plot_block_and_neighbors(coarse, 2, plotter=object(), show=False)
    with pytest.raises(ValueError, match="alpha"):
        plot_block_and_neighbors(
            coarse, 0, alpha=(1.0,), plotter=object(), show=False
        )


@pytest.mark.parametrize(
    "dimension,slice_index,tag",
    [(1, 1, 0), (2, 0, 2), (3, 1, 4)],
)
def test_plot_slice_selects_direction_tagged_faces_and_cell_values(
    dimension, slice_index, tag
):
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh
            self.kwargs = kwargs
            return "slice"

    grid = compute_geometry(cart_grid([3, 2, 2], [3.0, 2.0, 2.0]))
    data = 10.0+np.arange(grid.cells.num)
    cells, faces, owners = slice_entities(grid, slice_index, dimension)
    plotter = Plotter()
    result = plot_slice(
        grid,
        data,
        slice_index,
        dimension,
        plotter=plotter,
        show=False,
    )
    np.testing.assert_array_equal(result.cells, cells)
    np.testing.assert_array_equal(result.faces, faces)
    np.testing.assert_array_equal(result.values[:, 0], data[owners])
    np.testing.assert_array_equal(result.geometry.entity_ids, faces)
    np.testing.assert_array_equal(
        plotter.mesh.cell_data["face_data"], data[owners]
    )
    np.testing.assert_array_equal(result.view_vector,
                                  np.eye(3)[dimension-1])
    for cell, face in zip(owners, faces):
        lo, hi = grid.cells.facePos[cell:cell+2]
        local = np.flatnonzero(grid.cells.faces[lo:hi] == face)+lo
        assert tag in grid.cells.tags[local]


def test_plot_slice_skips_inactive_cartesian_cells():
    from pymrst.gridprocessing import remove_cells

    full = compute_geometry(cart_grid([3, 2, 1], [3.0, 2.0, 1.0]))
    grid = remove_cells(full, [1])
    data = np.arange(grid.cells.num, dtype=float)
    cells, faces, owners = slice_entities(grid, 1, 1)
    assert cells.size == 1
    assert np.all(cells != -1)
    assert faces.size == owners.size == 1


def test_plot_slice_validates_dimension_index_and_data_shape():
    grid = compute_geometry(cart_grid([2, 2, 1]))
    data = np.arange(grid.cells.num)
    with pytest.raises(ValueError, match="dimension"):
        plot_slice(grid, data, 0, 4, plotter=object(), show=False)
    with pytest.raises(IndexError, match="slice_index"):
        plot_slice(grid, data, 2, 1, plotter=object(), show=False)
    with pytest.raises(ValueError, match="expected"):
        plot_slice(grid, data[:-1], 0, 1, plotter=object(), show=False)


def test_interpolate_data_preserves_affine_fields_inside_and_outside_hull():
    grid = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
    points = grid.nodes.coords
    data = 3.0*points[:, 0]-2.0*points[:, 1]+5.0
    query = np.vstack([grid.cells.centroids, [3.0, -1.0]])
    result = interpolate_data(grid, data, xq=query, extrapolation="linear")
    expected = 3.0*query[:, 0]-2.0*query[:, 1]+5.0
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)


def test_interpolate_data_infers_cells_and_supports_multiple_columns():
    grid = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
    centers = grid.cells.centroids
    data = np.column_stack([
        centers[:, 0]+centers[:, 1],
        2.0*centers[:, 0]-centers[:, 1],
    ])
    result = interpolate_data(grid, data, xq="nodes")
    nodes = grid.nodes.coords
    expected = np.column_stack([
        nodes[:, 0]+nodes[:, 1],
        2.0*nodes[:, 0]-nodes[:, 1],
    ])
    np.testing.assert_allclose(result, expected, rtol=0, atol=1e-12)


def test_interpolate_data_nearest_and_no_extrapolation():
    grid = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
    points = grid.cells.centroids
    data = np.arange(grid.cells.num, dtype=float)
    query = np.array([[0.6, 0.6], [10.0, 10.0]])
    nearest = interpolate_data(
        grid, data, xq=query, interpolation="nearest"
    )
    assert nearest[0] == data[0]
    linear = interpolate_data(
        grid, data, xq=query, extrapolation="none"
    )
    assert np.isnan(linear[1])


def test_interpolate_data_validates_coordinates_queries_and_methods():
    grid = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
    with pytest.raises(ValueError, match="row count"):
        interpolate_data(grid, np.ones(3))
    with pytest.raises(ValueError, match="source coordinates"):
        interpolate_data(grid, np.ones(3), x=np.ones((3, 3)))
    with pytest.raises(ValueError, match="query"):
        interpolate_data(grid, np.ones(grid.cells.num), xq="bad")
    with pytest.raises(ValueError, match="interpolation"):
        interpolate_data(
            grid,
            np.ones(grid.cells.num),
            interpolation="cubic-spline",
        )


def test_surface_geometry_lifts_selected_cells_with_nodal_data():
    grid = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
    nodal = 2.0*grid.nodes.coords[:, 0]-grid.nodes.coords[:, 1]
    geometry, values = surface_geometry(grid, nodal, cells=[1, 4])
    np.testing.assert_array_equal(geometry.entity_ids, [1, 4])
    np.testing.assert_array_equal(values, nodal)
    np.testing.assert_allclose(
        geometry.vertices[:, 2], nodal[geometry.node_ids]
    )
    assert all(loop.size == 4 for loop in geometry.connectivity)


def test_unstructured_surf_interpolates_cell_data_and_renders_points():
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh
            self.kwargs = kwargs
            return "surface"

    grid = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
    cell_data = grid.cells.centroids[:, 0]+grid.cells.centroids[:, 1]
    plotter = Plotter()
    result = unstructured_surf(
        grid,
        cell_data,
        cells=np.array([True, False, False, True, False,
                        False, False, False, True]),
        plotter=plotter,
        show=False,
    )
    expected = grid.nodes.coords[:, 0]+grid.nodes.coords[:, 1]
    np.testing.assert_allclose(result.node_values, expected, atol=1e-12)
    np.testing.assert_array_equal(result.geometry.entity_ids, [0, 3, 8])
    np.testing.assert_allclose(
        plotter.mesh.point_data["surface_data"],
        expected[result.geometry.node_ids],
        rtol=0,
        atol=1e-12,
    )
    assert plotter.kwargs["preference"] == "point"
    assert result.actor == "surface"


def test_unstructured_surf_validates_dimension_and_scalar_data():
    grid3 = compute_geometry(cart_grid([1, 1, 1]))
    with pytest.raises(ValueError, match="2-D"):
        surface_geometry(grid3, np.ones(grid3.nodes.num))
    grid2 = compute_geometry(cart_grid([1, 1]))
    with pytest.raises(ValueError, match="scalar"):
        surface_geometry(grid2, np.ones((grid2.nodes.num, 3)))


def test_contour_grid_infers_sampling_dimensions_and_affine_values():
    grid = compute_geometry(cart_grid([4, 3], [8.0, 3.0]))
    cell_data = 2.0*grid.cells.centroids[:, 0]-grid.cells.centroids[:, 1]
    x, y, values, dims = contour_grid(grid, cell_data)
    assert dims == (4, 3)
    assert x.shape == y.shape == values.shape == dims
    np.testing.assert_allclose(values, 2.0*x-y, rtol=0, atol=1e-11)


def test_unstructured_contour_supports_explicit_levels_and_fill():
    import matplotlib.pyplot as plt

    grid = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
    nodal = grid.nodes.coords[:, 0]+grid.nodes.coords[:, 1]
    figure, axes = plt.subplots()
    try:
        result = unstructured_contour(
            grid,
            nodal,
            levels=[1.0, 2.0, 3.0, 4.0],
            cart_dims=(12, 10),
            fill=True,
            axes=axes,
        )
        assert result.cart_dims == (12, 10)
        assert result.filled is True
        np.testing.assert_allclose(result.levels, [1.0, 2.0, 3.0, 4.0])
        assert result.contour_set is not None
    finally:
        plt.close(figure)


def test_unstructured_contour_validates_grid_data_and_dimensions():
    grid3 = compute_geometry(cart_grid([1, 1, 1]))
    with pytest.raises(ValueError, match="2-D"):
        contour_grid(grid3, np.ones(grid3.cells.num))
    grid2 = compute_geometry(cart_grid([2, 2]))
    with pytest.raises(ValueError, match="scalar"):
        contour_grid(grid2, np.ones((grid2.nodes.num, 3)))
    with pytest.raises(ValueError, match="cart_dims"):
        contour_grid(grid2, np.ones(grid2.cells.num), cart_dims=(1, 3))


def test_contour_face_sets_match_discrete_band_and_threshold_logic():
    grid = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
    values = np.array([0.0, 0.75, 2.0])
    levels, sets = contour_face_sets(grid, values, 2)
    np.testing.assert_allclose(levels, [0.0, 1.0, 2.0])
    assert len(sets) == 2
    boundary = boundary_faces(grid)
    expected_low = boundary[np.any(
        np.isin(grid.faces.neighbors[boundary], [0, 1]), axis=1
    )]
    expected_high = boundary[np.any(
        grid.faces.neighbors[boundary] == 2, axis=1
    )]
    np.testing.assert_array_equal(sets[0][0], expected_low)
    np.testing.assert_array_equal(sets[1][0], expected_high)
    interface = np.flatnonzero(
        np.all(grid.faces.neighbors >= 0, axis=1)
        & np.any(grid.faces.neighbors == 1, axis=1)
        & np.any(grid.faces.neighbors == 2, axis=1)
    )
    np.testing.assert_array_equal(sets[0][1], interface)


def test_plot_contours_returns_all_band_actors_and_faces():
    class Plotter:
        def __init__(self):
            self.calls = []

        def add_mesh(self, mesh, **kwargs):
            self.calls.append((mesh, kwargs))
            return len(self.calls)

    grid = compute_geometry(cart_grid([3, 2, 1], [3.0, 2.0, 1.0]))
    values = np.arange(grid.cells.num, dtype=float)
    plotter = Plotter()
    result = plot_contours(
        grid, values, 3, plotter=plotter, show=False
    )
    expected_levels, expected_sets = contour_face_sets(grid, values, 3)
    np.testing.assert_array_equal(result.levels, expected_levels)
    assert len(result.bands) == 3
    for band, (exterior, internal) in zip(result.bands, expected_sets):
        np.testing.assert_array_equal(band.exterior_faces, exterior)
        np.testing.assert_array_equal(band.internal_faces, internal)
    assert result.plotter is plotter


def test_contour_face_sets_validates_grid_values_and_count():
    grid2 = compute_geometry(cart_grid([2, 2]))
    with pytest.raises(ValueError, match="3-D"):
        contour_face_sets(grid2, np.ones(grid2.cells.num), 2)
    grid3 = compute_geometry(cart_grid([1, 1, 1]))
    with pytest.raises(ValueError, match="expected"):
        contour_face_sets(grid3, [], 2)
    with pytest.raises(ValueError, match="positive"):
        contour_face_sets(grid3, [1.0], 0)


def test_histogram_colorbar_data_matches_matlab_center_bins():
    values = np.array([-2.0, -0.6, -0.4, 0.0, 0.4, 0.6, 2.0])
    counts, centers, limits = histogram_colorbar_data(
        values, [-1.0, 1.0], bins=2
    )
    np.testing.assert_allclose(centers, [-1.0, 0.0, 1.0])
    np.testing.assert_array_equal(counts, [2, 3, 2])
    assert limits == (-1.0, 1.0)


def test_colorbar_hist_renders_headlessly_in_all_supported_orientations():
    import matplotlib.pyplot as plt

    values = np.geomspace(1.0, 1000.0, 30)
    for location in ("south", "north", "east", "west"):
        figure, axes = plt.subplots()
        try:
            result = colorbar_hist(
                values,
                [1.0, 1000.0],
                location,
                bins=6,
                logarithmic=True,
                axes=axes,
            )
            # MRST honours the bin count only without the log flag
            # (`nargin == 4`); with it, 51 centres.
            assert result.counts.size == 51
            assert result.location == location
            assert result.logarithmic is True
            assert result.colorbar is not None
            assert result.histogram_axes is not None
        finally:
            plt.close(figure)


def test_mrst_colorbar_accepts_values_and_infers_artist_data():
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots()
    try:
        image = axes.imshow(np.arange(9).reshape(3, 3))
        assert image is not None
        inferred = mrst_colorbar(axes, bins=5)
        np.testing.assert_array_equal(inferred.counts.sum(), 9)
    finally:
        plt.close(figure)


def test_mrst_colorbar_north_is_physically_above_axes_and_distinct_from_south():
    import matplotlib.pyplot as plt

    values = np.linspace(0.0, 1.0, 30)
    north_figure, north_axes = plt.subplots()
    south_figure, south_axes = plt.subplots()
    try:
        north = mrst_colorbar(
            north_axes, values, location="northoutside", bins=5
        )
        south = mrst_colorbar(
            south_axes, values, location="southoutside", bins=5
        )
        north_hist = north.histogram_axes.get_position()
        north_color = north.colorbar.ax.get_position()
        axes_box = north_axes.get_position()
        south_hist = south.histogram_axes.get_position()
        south_color = south.colorbar.ax.get_position()

        assert north.location == "north"
        assert north_hist.y0 >= axes_box.y1
        assert north_color.y0 > north_hist.y1
        assert north.histogram_axes.yaxis_inverted()
        assert north.colorbar.ax.xaxis.get_ticks_position() == "top"
        assert south_hist.y1 <= south_axes.get_position().y0
        assert south_color.y0 < south_hist.y0
    finally:
        plt.close(north_figure)
        plt.close(south_figure)


def test_histogram_colorbar_validates_limits_log_data_and_location():
    with pytest.raises(ValueError, match="increasing"):
        histogram_colorbar_data([1.0], [1.0, 1.0])
    with pytest.raises(ValueError, match="positive"):
        histogram_colorbar_data([-1.0, 1.0], [0.1, 2.0], logarithmic=True)
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots()
    try:
        with pytest.raises(ValueError, match="location"):
            colorbar_hist([1.0], [0.0, 2.0], "inside", axes=axes)
    finally:
        plt.close(figure)


def test_volume_samples_extrudes_affine_cell_data_to_domain_boundary():
    grid = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))
    values = grid.cells.centroids.sum(axis=1)
    samples = volume_samples(grid, values, extrude_faces=True)
    assert samples.values.shape == (5, 5, 5)
    center_indices = np.ix_([1, 3], [1, 3], [1, 3])
    np.testing.assert_allclose(
        samples.values[center_indices],
        (samples.x+samples.y+samples.z)[center_indices],
        rtol=0,
        atol=1e-11,
    )
    assert samples.source_values.size == (
        grid.cells.num+boundary_faces(grid).size
    )


def test_volume_histogram_centers_follow_matlab_scalar_bin_rule():
    values = np.array([1.5, 2.5, 2.5, 3.5, 4.5])
    np.testing.assert_allclose(_histogram_centers(values, 3), [2.0, 3.0, 4.0])


def test_volume_samples_respects_mesh_bounds_and_reuses_interpolant():
    grid = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))
    values = grid.cells.centroids[:, 0]
    first = volume_samples(
        grid, values, mesh=(3, 4, 5), extrude_faces=False
    )
    assert first.values.shape == (5, 4, 6)
    np.testing.assert_allclose(first.axes[0], np.linspace(0.5, 1.5, 4))
    reused = volume_samples(
        grid,
        values,
        mesh=(3, 4, 5),
        extrude_faces=False,
        interpolant=first.values,
    )
    np.testing.assert_array_equal(reused.values, first.values)


def test_plot_grid_volumes_builds_isosurfaces_and_alpha_progression():
    class Plotter:
        def __init__(self):
            self.calls = []

        def add_mesh(self, mesh, **kwargs):
            self.calls.append((mesh, kwargs))
            return len(self.calls)

    grid = compute_geometry(cart_grid([3, 3, 3], [3.0, 3.0, 3.0]))
    center = np.array([1.5, 1.5, 1.5])
    values = np.linalg.norm(grid.cells.centroids-center, axis=1)
    plotter = Plotter()
    result = plot_grid_volumes(
        grid,
        values,
        binc=[0.75, 1.25],
        mesh=(8, 8, 8),
        basealpha=0.8,
        patchn=500,
        plotter=plotter,
        show=False,
    )
    np.testing.assert_allclose(result.isovalues, [0.75, 1.25])
    np.testing.assert_allclose(
        [band.opacity for band in result.bands], [0.2, 0.4]
    )
    assert all(band.face_count <= 250 for band in result.bands)
    assert any(band.actor is not None for band in result.bands)


def test_plot_grid_volumes_accepts_mrst_min_max_and_callable_cmap():
    class Plotter:
        def add_mesh(self, mesh, **kwargs):
            return "actor"

    grid = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))
    values = grid.cells.centroids.sum(axis=1)
    result = plot_grid_volumes(
        grid,
        values,
        N=2,
        mesh=(3, 3, 3),
        cmap=lambda count: np.tile([0.1, 0.2, 0.3], (count, 1)),
        plotter=Plotter(),
        show=False,
        **{"min": 2.0, "max": 4.0},
    )
    assert len(result.bands) == 2
    assert all(band.color == (0.1, 0.2, 0.3) for band in result.bands)


def test_plot_grid_volumes_validates_inputs_and_interpolant_shape():
    grid2 = compute_geometry(cart_grid([2, 2]))
    with pytest.raises(ValueError, match="3-D"):
        volume_samples(grid2, np.ones(grid2.cells.num))
    grid3 = compute_geometry(cart_grid([1, 1, 1]))
    with pytest.raises(ValueError, match="expected"):
        volume_samples(grid3, [])
    with pytest.raises(ValueError, match="mesh"):
        volume_samples(grid3, [1.0], mesh=(0, 2, 2))
    with pytest.raises(ValueError, match="shape"):
        volume_samples(grid3, [1.0], interpolant=np.zeros((2, 2, 2)))
    with pytest.raises(ValueError, match="remain"):
        plot_grid_volumes(
            grid3,
            [1.0],
            minimum=2.0,
            plotter=object(),
            show=False,
        )
