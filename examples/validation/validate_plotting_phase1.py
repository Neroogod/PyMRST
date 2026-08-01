"""Compare Phase-1 plotting topology against MATLAB MRST."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import scipy.io

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.upscaling.coarsegrid import (
    generate_coarse_grid,
    partition_cart_grid,
)
from pymrst.plotting.boundary_faces import boundary_faces
from pymrst.plotting.private.coarse_faces import expand_coarse_faces
from pymrst.plotting.private.plot_face_outline import face_outline_geometry
from pymrst.plotting.private.plot_patches import build_patch_geometry
from pymrst.plotting.plot_cell_data import plot_cell_data
from pymrst.plotting.plot_node_data import plot_node_data
from pymrst.plotting.plot_grid import plot_grid
from pymrst.plotting.plot_well import plot_well
from pymrst.params.wells_and_bc import Well
from pymrst.plotting.plot_faults import plot_faults
from pymrst.gridprocessing import process_faults
from pymrst.plotting.outline_coarse_grid import coarse_partition_faces
from pymrst.plotting.plot_block_and_neighbors import (
    plot_block_and_neighbors,
)
from pymrst.plotting.plot_slice import slice_entities
from pymrst.plotting.interpolate_data import interpolate_data
from pymrst.plotting.unstructured_surf import surface_geometry
from pymrst.plotting.unstructured_contour import contour_grid
from pymrst.plotting.plot_contours import contour_face_sets
from pymrst.plotting.colorbar_hist import histogram_colorbar_data
from pymrst.plotting.plot_grid_volumes import (
    _histogram_centers,
    volume_samples,
)


HERE = Path(__file__).parent


def _flat(reference, name, *, index=False):
    values = np.asarray(reference[name]).ravel()
    return values.astype(np.int64)-(1 if index else 0)


def _group_signature(faces, groups):
    """Canonicalize grouped faces independently of coarse-face ordering."""
    faces = np.asarray(faces, dtype=np.int64).ravel()
    groups = np.asarray(groups, dtype=np.int64).ravel()
    return sorted(
        tuple(sorted(map(int, faces[groups == group])))
        for group in np.unique(groups)
    )


def main() -> int:
    path = HERE / "ref_plotting_phase1.mat"
    if not path.exists():
        print("ref_plotting_phase1.mat missing "
              "(run dump_plotting_phase1_reference.m)")
        return 1
    reference = scipy.io.loadmat(path, squeeze_me=True)
    failed = 0

    def exact(name, got, expected):
        nonlocal failed
        if isinstance(got, list) and isinstance(expected, list):
            ok = got == expected
        else:
            got = np.asarray(got)
            expected = np.asarray(expected)
            ok = got.shape == expected.shape and np.array_equal(got, expected)
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
        failed += 0 if ok else 1

    g2 = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
    faces, cells = boundary_faces(g2, [0, 1], with_cells=True)
    exact("2-D subset boundary faces", faces,
          _flat(reference, "b2_faces", index=True))
    exact("2-D subset owning cells", cells,
          _flat(reference, "b2_cells", index=True))

    g3 = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
    boundary = boundary_faces(g3)
    exact("3-D boundary faces", boundary,
          _flat(reference, "g3_boundary", index=True))
    geometry = build_patch_geometry(g3, boundary)
    face_nodes = np.concatenate(geometry.connectivity)
    face_pos = np.cumsum([0, *map(len, geometry.connectivity)])
    exact("compact node IDs", geometry.node_ids,
          _flat(reference, "g3_node_ids", index=True))
    exact("compact face positions", face_pos,
          _flat(reference, "g3_face_pos", index=True))
    exact("compact face nodes", face_nodes,
          _flat(reference, "g3_face_nodes", index=True))
    vertices = np.asarray(reference["g3_vertices"], dtype=float)
    close = np.allclose(geometry.vertices, vertices, rtol=0, atol=1e-14)
    print(f"  {'ok  ' if close else 'FAIL'} compact vertices")
    failed += 0 if close else 1

    bottom = _flat(reference, "g3_bottom", index=True)
    outline = face_outline_geometry(g3, bottom)
    global_edges = np.asarray([
        geometry_ids
        for geometry_ids in (
            outline.node_ids[edge] for edge in outline.connectivity
        )
    ], dtype=np.int64)
    global_edges = np.asarray(sorted(map(tuple, np.sort(global_edges, axis=1))))
    expected_edges = np.asarray(
        reference["g3_outline_edges"], dtype=np.int64
    ).reshape(-1, 2)-1
    exact("collection outline", global_edges, expected_edges)

    fine = compute_geometry(cart_grid([4, 2, 1], [4.0, 2.0, 1.0]))
    coarse = generate_coarse_grid(
        fine, partition_cart_grid([4, 2, 1], [2, 1, 1])
    )
    coarse_boundary = boundary_faces(coarse.g)
    _, fine_faces, groups = expand_coarse_faces(coarse, coarse_boundary)
    exact(
        "coarse boundary face count",
        np.asarray([coarse_boundary.size]),
        np.asarray([_flat(reference, "cg_boundary", index=True).size]),
    )
    exact(
        "coarse grouped fine-face topology",
        _group_signature(fine_faces, groups),
        _group_signature(
            _flat(reference, "cg_fine_faces", index=True),
            _flat(reference, "cg_groups", index=True),
        ),
    )

    class CapturePlotter:
        def add_mesh(self, mesh, **kwargs):
            self.mesh = mesh
            self.kwargs = kwargs
            return "actor"

    data_grid = compute_geometry(cart_grid([3, 1, 1], [3.0, 1.0, 1.0]))
    capture = CapturePlotter()
    cell_result = plot_cell_data(
        data_grid,
        np.asarray([10.0, 20.0]),
        np.asarray([1, 2]),
        plotter=capture,
        show=False,
    )
    exact(
        "cell-data boundary faces",
        cell_result.geometry.entity_ids,
        _flat(reference, "cell_data_faces", index=True),
    )
    exact(
        "cell-data owning values",
        capture.mesh.cell_data["face_data"],
        _flat(reference, "cell_data_values"),
    )

    node_capture = CapturePlotter()
    node_result = plot_node_data(
        data_grid,
        np.arange(data_grid.nodes.num, dtype=float),
        [1],
        plotter=node_capture,
        show=False,
    )
    exact(
        "node-data compact node IDs",
        node_result.geometry.node_ids,
        _flat(reference, "node_data_ids", index=True),
    )

    line_grid = compute_geometry(cart_grid([4], [8.0]))
    line_cells = _flat(reference, "line_cells", index=True)
    line_capture = CapturePlotter()
    line_data_result = plot_cell_data(
        line_grid,
        _flat(reference, "line_values"),
        line_cells,
        plotter=line_capture,
        show=False,
    )
    exact(
        "1-D cell-data line geometry",
        line_data_result.geometry.vertices,
        np.asarray(reference["line_cell_data_xy"], dtype=float),
    )
    grid_line_result = plot_grid(
        line_grid,
        line_cells,
        plotter=CapturePlotter(),
        show=False,
    )
    exact(
        "1-D grid line geometry",
        grid_line_result.geometry.vertices,
        np.asarray(reference["line_grid_xy"], dtype=float),
    )

    well_grid = compute_geometry(cart_grid([1, 1, 3], [1.0, 1.0, 3.0]))
    well_cells = _flat(reference, "well_cells", index=True)
    well = Well(
        cells=well_cells,
        type="rate",
        val=1.0,
        r=np.full(well_cells.size, 0.1),
        dir="z",
        WI=np.ones(well_cells.size),
        dZ=np.zeros(well_cells.size),
        name="W1",
        compi=np.asarray([1.0, 0.0]),
        refDepth=0.0,
        sign=1,
    )
    well_capture = CapturePlotter()
    well_capture.add_point_labels = lambda *args, **kwargs: "label"
    well_result = plot_well(
        well_grid,
        well,
        well_capture,
        height=2.0,
        fontsize=0,
    )
    well_geometry = well_result.geometry[0]
    exact(
        "well completion points",
        well_geometry.completion_points,
        np.asarray(reference["well_completions"], dtype=float),
    )
    exact(
        "well head",
        well_geometry.head,
        np.asarray(reference["well_head"], dtype=float).ravel(),
    )
    exact(
        "well heel",
        well_geometry.heel,
        np.asarray(reference["well_heel"], dtype=float).ravel(),
    )
    exact(
        "well trajectory",
        well_geometry.trajectory,
        np.asarray(reference["well_trajectory"], dtype=float),
    )
    close = np.isclose(
        well_geometry.radius,
        float(np.asarray(reference["well_radius"]).ravel()[0]),
        rtol=0,
        atol=1e-14,
    )
    print(f"  {'ok  ' if close else 'FAIL'} well radius")
    failed += 0 if close else 1

    fault_grid = compute_geometry(cart_grid([3, 3, 2]))
    faults, _ = process_faults(fault_grid, {
        "FAULTS": [
            ["BOUND", 1, 1, 1, 3, 1, 2, "X-"],
            ["MAIN", 2, 2, 1, 2, 1, 2, "X+"],
        ],
    })
    fault_capture = CapturePlotter()
    fault_capture.add_point_labels = lambda *args, **kwargs: "labels"
    fault_result = plot_faults(
        fault_grid,
        faults,
        height=2.5,
        plotter=fault_capture,
        show=False,
    )
    exact(
        "fault face IDs",
        fault_result.faces.geometry.entity_ids,
        _flat(reference, "fault_faces", index=True),
    )
    exact(
        "fault face groups",
        fault_result.face_groups,
        _flat(reference, "fault_groups", index=True),
    )
    label_close = np.allclose(
        fault_result.label_positions,
        np.asarray(reference["fault_label_positions"], dtype=float),
        rtol=0,
        atol=1e-14,
    )
    print(f"  {'ok  ' if label_close else 'FAIL'} fault label positions")
    failed += 0 if label_close else 1

    outline_grid2 = compute_geometry(cart_grid([4, 2], [4.0, 2.0]))
    outline_partition2 = partition_cart_grid([4, 2], [2, 1])
    exact(
        "2-D coarse outline faces",
        coarse_partition_faces(outline_grid2, outline_partition2),
        _flat(reference, "outline_faces2", index=True),
    )
    outline_grid3 = compute_geometry(
        cart_grid([4, 2, 1], [4.0, 2.0, 1.0])
    )
    outline_partition3 = partition_cart_grid([4, 2, 1], [2, 1, 1])
    exact(
        "3-D coarse outline faces",
        coarse_partition_faces(outline_grid3, outline_partition3),
        _flat(reference, "outline_faces3", index=True),
    )

    block_fine = compute_geometry(
        cart_grid([6, 2, 1], [6.0, 2.0, 1.0])
    )
    block_partition = partition_cart_grid([6, 2, 1], [3, 1, 1])
    block_coarse = generate_coarse_grid(block_fine, block_partition)
    block_id = int(np.asarray(reference["block_id"]).ravel()[0])-1
    block_capture = CapturePlotter()
    block_result = plot_block_and_neighbors(
        block_coarse,
        block_id,
        plot_faults=(False, False),
        alpha=(1.0, 1.0),
        plotter=block_capture,
        show=False,
    )
    exact(
        "coarse block neighbors",
        block_result.neighbors,
        _flat(reference, "block_neighbors", index=True),
    )
    exact(
        "coarse block boundary faces",
        block_result.block_faces,
        _flat(reference, "block_boundary_faces", index=True),
    )
    neighbor_pos = _flat(reference, "neighbor_face_pos", index=True)
    neighbor_faces = _flat(
        reference, "neighbor_boundary_faces", index=True
    )
    for ordinal, got in enumerate(block_result.neighbor_faces):
        exact(
            f"coarse neighbor {ordinal} boundary faces",
            got,
            neighbor_faces[neighbor_pos[ordinal]:neighbor_pos[ordinal+1]],
        )

    slice_grid = compute_geometry(
        cart_grid([3, 2, 2], [3.0, 2.0, 2.0])
    )
    slice_data = 10.0+np.arange(slice_grid.cells.num)
    for dimension, index in ((1, 1), (2, 0), (3, 1)):
        cells, faces, owners = slice_entities(slice_grid, index, dimension)
        exact(
            f"slice dimension {dimension} cells",
            cells,
            _flat(reference, f"slice_cells{dimension}", index=True),
        )
        exact(
            f"slice dimension {dimension} faces",
            faces,
            _flat(reference, f"slice_faces{dimension}", index=True),
        )
        exact(
            f"slice dimension {dimension} values",
            slice_data[owners],
            _flat(reference, f"slice_values{dimension}"),
        )

    interpolation_grid = compute_geometry(
        cart_grid([2, 2], [2.0, 2.0])
    )
    affine_result = interpolate_data(
        interpolation_grid,
        np.asarray(reference["interp_node_data"], dtype=float).ravel(),
        xq=np.asarray(reference["interp_query"], dtype=float),
        interpolation="linear",
        extrapolation="linear",
    )
    affine_close = np.allclose(
        affine_result,
        np.asarray(reference["interp_affine"], dtype=float).ravel(),
        rtol=0,
        atol=1e-12,
    )
    print(f"  {'ok  ' if affine_close else 'FAIL'} affine interpolation")
    failed += 0 if affine_close else 1
    cell_node_result = interpolate_data(
        interpolation_grid,
        np.asarray(reference["interp_cell_data"], dtype=float).ravel(),
        xq="nodes",
        interpolation="linear",
        extrapolation="linear",
    )
    cell_node_close = np.allclose(
        cell_node_result,
        np.asarray(reference["interp_cells_to_nodes"], dtype=float).ravel(),
        rtol=0,
        atol=1e-12,
    )
    print(f"  {'ok  ' if cell_node_close else 'FAIL'} cell-to-node interpolation")
    failed += 0 if cell_node_close else 1

    surface_grid = compute_geometry(cart_grid([3, 3], [3.0, 3.0]))
    surface, surface_nodes = surface_geometry(
        surface_grid,
        np.asarray(reference["surface_cell_data"], dtype=float).ravel(),
        _flat(reference, "surface_cells", index=True),
    )
    surface_values_close = np.allclose(
        surface_nodes,
        np.asarray(reference["surface_node_values"], dtype=float).ravel(),
        rtol=0,
        atol=1e-12,
    )
    print(f"  {'ok  ' if surface_values_close else 'FAIL'} surface nodal values")
    failed += 0 if surface_values_close else 1
    exact(
        "surface compact node IDs",
        surface.node_ids,
        _flat(reference, "surface_node_ids", index=True),
    )
    surface_nodes_flat = np.concatenate(surface.connectivity)
    exact(
        "surface compact cell nodes",
        surface_nodes_flat,
        _flat(reference, "surface_cell_nodes", index=True),
    )
    surface_pos = np.cumsum([0, *map(len, surface.connectivity)])
    exact(
        "surface compact cell positions",
        surface_pos,
        _flat(reference, "surface_cell_pos", index=True),
    )
    surface_vertices_close = np.allclose(
        surface.vertices,
        np.asarray(reference["surface_vertices"], dtype=float),
        rtol=0,
        atol=1e-12,
    )
    print(f"  {'ok  ' if surface_vertices_close else 'FAIL'} surface vertices")
    failed += 0 if surface_vertices_close else 1

    contour_source_grid = compute_geometry(
        cart_grid([4, 3], [8.0, 3.0])
    )
    contour_x, contour_y, contour_values, contour_dims = contour_grid(
        contour_source_grid,
        np.asarray(reference["contour_cell_data"], dtype=float).ravel(),
    )
    exact(
        "contour inferred dimensions",
        contour_dims,
        tuple(np.asarray(reference["contour_dims"], dtype=int).ravel()),
    )
    for name, got in (
        ("x", contour_x),
        ("y", contour_y),
        ("values", contour_values),
    ):
        close = np.allclose(
            got,
            np.asarray(reference[f"contour_{name}"], dtype=float),
            rtol=0,
            atol=1e-12,
        )
        print(f"  {'ok  ' if close else 'FAIL'} contour {name}")
        failed += 0 if close else 1

    discrete_grid = compute_geometry(
        cart_grid([3, 2, 1], [3.0, 2.0, 1.0])
    )
    discrete_data = np.asarray(
        reference["plot_contour_data"], dtype=float
    ).ravel()
    discrete_count = int(
        np.asarray(reference["plot_contour_count"]).ravel()[0]
    )
    discrete_levels, discrete_sets = contour_face_sets(
        discrete_grid, discrete_data, discrete_count
    )
    levels_close = np.allclose(
        discrete_levels,
        np.asarray(reference["plot_contour_levels"], dtype=float).ravel(),
        rtol=0,
        atol=1e-14,
    )
    print(f"  {'ok  ' if levels_close else 'FAIL'} discrete contour levels")
    failed += 0 if levels_close else 1
    ext_pos = _flat(reference, "plot_contour_ext_pos", index=True)
    ext_faces = _flat(reference, "plot_contour_ext_faces", index=True)
    int_pos = _flat(reference, "plot_contour_int_pos", index=True)
    int_faces = _flat(reference, "plot_contour_int_faces", index=True)
    for ordinal, (exterior, internal) in enumerate(discrete_sets):
        exact(
            f"discrete contour {ordinal} exterior faces",
            exterior,
            ext_faces[ext_pos[ordinal]:ext_pos[ordinal+1]],
        )
        exact(
            f"discrete contour {ordinal} internal faces",
            internal,
            int_faces[int_pos[ordinal]:int_pos[ordinal+1]],
        )

    histogram_counts, histogram_centers, _ = histogram_colorbar_data(
        np.asarray(reference["hist_values"], dtype=float).ravel(),
        np.asarray(reference["hist_limits"], dtype=float).ravel(),
        bins=2,
    )
    exact(
        "histogram linear counts",
        histogram_counts,
        _flat(reference, "hist_counts"),
    )
    histogram_centers_close = np.allclose(
        histogram_centers,
        np.asarray(reference["hist_centers"], dtype=float).ravel(),
        rtol=0,
        atol=1e-14,
    )
    print(f"  {'ok  ' if histogram_centers_close else 'FAIL'} histogram linear centers")
    failed += 0 if histogram_centers_close else 1
    log_counts, log_centers, log_limits = histogram_colorbar_data(
        np.asarray(reference["hist_log_values"], dtype=float).ravel(),
        np.asarray(reference["hist_log_limits"], dtype=float).ravel(),
        bins=6,
        logarithmic=True,
    )
    exact("histogram log counts", log_counts,
          _flat(reference, "hist_log_counts"))
    log_centers_close = np.allclose(
        log_centers,
        np.asarray(reference["hist_log_centers"], dtype=float).ravel(),
        rtol=0,
        atol=1e-14,
    )
    print(f"  {'ok  ' if log_centers_close else 'FAIL'} histogram log centers")
    failed += 0 if log_centers_close else 1
    log_limits_close = np.allclose(log_limits, [0.0, 3.0], atol=1e-14)
    print(f"  {'ok  ' if log_limits_close else 'FAIL'} histogram log limits")
    failed += 0 if log_limits_close else 1

    volume_grid = compute_geometry(
        cart_grid([2, 2, 2], [2.0, 2.0, 2.0])
    )
    volume_data = np.asarray(
        reference["volume_cell_data"], dtype=float
    ).ravel()
    volume_result = volume_samples(
        volume_grid, volume_data, extrude_faces=True
    )
    exact(
        "volume boundary faces",
        boundary_faces(volume_grid),
        _flat(reference, "volume_boundary_faces", index=True),
    )
    source_close = np.allclose(
        volume_result.source_coordinates,
        np.asarray(reference["volume_source_coordinates"], dtype=float),
        rtol=0,
        atol=1e-14,
    )
    print(f"  {'ok  ' if source_close else 'FAIL'} volume source coordinates")
    failed += 0 if source_close else 1
    source_values_close = np.allclose(
        volume_result.source_values,
        np.asarray(reference["volume_source_values"], dtype=float).ravel(),
        rtol=0,
        atol=1e-14,
    )
    print(f"  {'ok  ' if source_values_close else 'FAIL'} volume source values")
    failed += 0 if source_values_close else 1
    for axis, name in zip(volume_result.axes, ("x", "y", "z")):
        close = np.allclose(
            axis,
            np.asarray(reference[f"volume_axis_{name}"], dtype=float).ravel(),
            rtol=0,
            atol=1e-14,
        )
        print(f"  {'ok  ' if close else 'FAIL'} volume axis {name}")
        failed += 0 if close else 1
    matlab_samples = np.asarray(reference["volume_samples_ref"], dtype=float)
    # Boundary-corner extrapolation depends on the Delaunay triangulation;
    # compare the shared interpolation domain exactly and test full-field
    # reuse separately.
    volume_sample_close = np.allclose(
        volume_result.values[1:4, 1:4, 1:4],
        matlab_samples[1:4, 1:4, 1:4],
        rtol=0,
        atol=1e-11,
    )
    print(f"  {'ok  ' if volume_sample_close else 'FAIL'} volume interior samples")
    failed += 0 if volume_sample_close else 1
    reused_volume = volume_samples(
        volume_grid,
        volume_data,
        extrude_faces=True,
        interpolant=matlab_samples,
    )
    exact("volume reused sampled field", reused_volume.values, matlab_samples)
    isovalues = _histogram_centers(volume_data, 3)
    isovalues_close = np.allclose(
        isovalues,
        np.asarray(reference["volume_isovalues"], dtype=float).ravel(),
        rtol=0,
        atol=1e-14,
    )
    print(f"  {'ok  ' if isovalues_close else 'FAIL'} volume isovalues")
    failed += 0 if isovalues_close else 1

    print("\nPlotting Phase-1 topology matches MRST!" if failed == 0
          else f"\n{failed} comparisons FAILED")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
