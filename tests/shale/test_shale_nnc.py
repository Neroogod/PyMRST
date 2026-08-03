import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.tetrahedral_grid import tetrahedral_grid
from pymrst.modules.hfm import (
    PlanarFracture,
    build_edfm_3d,
    project_planar_fracture_to_grid,
)
from pymrst.modules.shale import (
    ShaleFracturePlane,
    ShaleNNC,
    ShaleRock,
    assemble_shale_global_grid,
    fracture_matrix_shale_nncs_3d,
    fracture_fracture_shale_nncs_3d,
    edfm_shale_grid,
    projected_matrix_fracture_nncs_3d,
    setup_pedfm_operators_tpfa,
    setup_shale_edfm_operators_tpfa,
    transmissibility_multiplier_pedfm,
)
from pymrst.utils.adi import init_variables_adi, value


def _two_cell_grid():
    return compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))


def _face_fracture():
    return PlanarFracture(
        center=np.array([1.0, 0.5, 0.5]),
        normal=np.array([1.0, 0.0, 0.0]),
        axis1=np.array([0.0, 1.0, 0.0]),
        half_length1=0.5,
        half_length2=0.5,
    )


def test_transmissibility_multiplier_reduces_matching_face_area():
    grid = _two_cell_grid()
    multiplier = transmissibility_multiplier_pedfm(grid, [[0, 1]], [0.25])
    internal = np.flatnonzero(np.all(grid.faces.neighbors >= 0, axis=1))
    assert internal.size == 1
    assert multiplier[internal[0]] == pytest.approx(0.75)
    boundary = np.setdiff1d(np.arange(grid.faces.num), internal)
    assert np.allclose(multiplier[boundary], 1.0)


def test_transmissibility_multiplier_is_orientation_independent_and_accumulates():
    grid = _two_cell_grid()
    multiplier = transmissibility_multiplier_pedfm(
        grid, [[1, 0], [0, 1]], [0.2, 0.3]
    )
    internal = np.flatnonzero(np.all(grid.faces.neighbors >= 0, axis=1))[0]
    assert multiplier[internal] == pytest.approx(0.5)
    with pytest.raises(ValueError, match="not a matrix face"):
        transmissibility_multiplier_pedfm(grid, [[0, 4]], [0.1])


def test_hfm_face_fracture_populates_native_shale_nnc_metadata():
    grid = _two_cell_grid()
    fracture = _face_fracture()
    projection = project_planar_fracture_to_grid(grid, fracture, aperture=0.01)
    edfm = build_edfm_3d(
        grid,
        [fracture],
        aperture=0.01,
        perm_matrix=np.full(grid.cells.num, 1.0e-15),
        perm_frac=1.0e-10,
        poro_matrix=np.full(grid.cells.num, 0.1),
        poro_frac=0.5,
    )
    nnc = fracture_matrix_shale_nncs_3d(edfm, [fracture], [projection])
    assert nnc.cells.shape == (2, 2)
    assert set(nnc.cells[:, 0]) == {0, 1}
    assert np.unique(nnc.cells[:, 1]).size == 1
    assert nnc.type == ("fracmat boundary",) * 2
    raw = edfm.trans[np.asarray(edfm.conn_type) == "frac_matrix"]
    assert nnc.T[0] == pytest.approx(2.0 * raw[0])
    assert nnc.T[1] == pytest.approx(nnc.T[0])
    assert np.allclose(nnc.area, 1.0)
    assert np.allclose(nnc.normal, [[1.0, 0.0, 0.0]] * 2)
    assert np.all(nnc.T > 0)

    global_grid = assemble_shale_global_grid(edfm)
    assert global_grid.n_matrix == 2
    assert global_grid.n_fracture == 1
    assert global_grid.n_cells == 3
    assert np.array_equal(global_grid.rock.isMatrix, [True, True, False])
    assert global_grid.nnc.cells.shape[0] == 1


def test_edfm_shale_grid_builds_complete_native_pipeline_and_processing_metadata():
    grid = compute_geometry(cart_grid([3, 3, 3], [3.0, 3.0, 3.0]))
    plane = ShaleFracturePlane(
        points=np.array([
            [1.5, 1.15, 1.15], [1.5, 1.85, 1.15],
            [1.5, 1.85, 1.85], [1.5, 1.15, 1.85],
        ]),
        aperture=0.01, permeability=1e-10, porosity=0.5,
    )
    matrix_rock = ShaleRock(
        perm=np.full((grid.cells.num, 1), 1e-15),
        poro=np.full(grid.cells.num, 0.1),
        is_matrix=np.ones(grid.cells.num, dtype=bool),
        shale_mechanisms={"sorption": True},
    )
    global_grid, processed = edfm_shale_grid(grid, [plane], matrix_rock)
    assert global_grid.n_matrix == grid.cells.num
    assert global_grid.n_fracture == 1
    assert global_grid.nnc.cells.shape == (1, 2)
    assert global_grid.nnc.type == ("fracmat interior",)
    raw = global_grid.edfm.trans[
        np.asarray(global_grid.edfm.conn_type) == "frac_matrix"
    ]
    assert np.allclose(global_grid.nnc.T, 4.0 * raw)
    assert global_grid.edfm is not None
    assert len(global_grid.projections) == 1
    assert processed[0].metadata["processed"] is True
    assert processed[0].metadata["fracgrid"] == 0
    assert np.allclose(np.abs(processed[0].metadata["normal"]), [1, 0, 0])

    volumes = np.concatenate([grid.cells.volumes, global_grid.fracture_volumes])
    operators = setup_shale_edfm_operators_tpfa(
        grid, global_grid.rock, np.ones(grid.faces.num), global_grid.nnc,
        cell_volumes=volumes,
    )
    assert operators.N.shape[0] == (
        np.count_nonzero(np.all(grid.faces.neighbors >= 0, axis=1)) + 1
    )
    assert operators.pv.size == global_grid.n_cells
    assert operators.grain_volume[-1] == 0.0

    merged = fracture_matrix_shale_nncs_3d(
        global_grid.edfm, global_grid.fractures, global_grid.projections,
        existing=global_grid.nnc,
    )
    assert merged.cells.shape == global_grid.nnc.cells.shape
    assert np.allclose(merged.T, global_grid.nnc.T)


def test_edfm_shale_grid_rejects_already_processed_selection():
    grid = compute_geometry(cart_grid([1, 1, 1]))
    plane = ShaleFracturePlane(
        points=np.array([
            [.5, .2, .2], [.5, .8, .2], [.5, .8, .8], [.5, .2, .8],
        ]), aperture=.01, permeability=1e-10, porosity=.5,
        metadata={"processed": True},
    )
    rock = ShaleRock(
        perm=np.full((1, 1), 1e-15), poro=np.array([.1]),
        is_matrix=np.array([True]),
    )
    with pytest.raises(ValueError, match="no unprocessed"):
        edfm_shale_grid(grid, [plane], rock)


def test_edfm_shale_grid_preserves_heterogeneous_fracture_properties():
    grid = compute_geometry(cart_grid([3, 3, 3], [3.0, 3.0, 3.0]))
    planes = [
        ShaleFracturePlane(
            points=np.array([
                [x, 1.2, 1.2], [x, 1.8, 1.2],
                [x, 1.8, 1.8], [x, 1.2, 1.8],
            ]),
            aperture=aperture, permeability=permeability, porosity=porosity,
        )
        for x, aperture, permeability, porosity in [
            (0.5, 0.01, 1e-10, 0.4), (2.5, 0.02, 3e-10, 0.7)
        ]
    ]
    rock = ShaleRock(
        perm=np.full((grid.cells.num, 1), 1e-15),
        poro=np.full(grid.cells.num, 0.1),
        is_matrix=np.ones(grid.cells.num, dtype=bool),
    )
    global_grid, _ = edfm_shale_grid(grid, planes, rock)
    assert global_grid.n_fracture == 2
    fracture_slice = slice(global_grid.n_matrix, None)
    assert np.allclose(global_grid.rock.perm[fracture_slice, 0], [1e-10, 3e-10])
    assert np.allclose(global_grid.rock.poro[fracture_slice], [0.4, 0.7])
    assert global_grid.fracture_volumes[1] == pytest.approx(
        2.0 * global_grid.fracture_volumes[0]
    )


def test_edfm_shale_grid_incrementally_adds_unprocessed_planes_with_stable_order():
    grid = compute_geometry(cart_grid([3, 3, 3], [3.0, 3.0, 3.0]))
    planes = [
        ShaleFracturePlane(
            points=np.array([
                [x, 1.2, 1.2], [x, 1.8, 1.2],
                [x, 1.8, 1.8], [x, 1.2, 1.8],
            ]), aperture=.01, permeability=1e-10, porosity=.5,
        )
        for x in (.5, 2.5)
    ]
    rock = ShaleRock(
        perm=np.full((grid.cells.num, 1), 1e-15),
        poro=np.full(grid.cells.num, .1),
        is_matrix=np.ones(grid.cells.num, dtype=bool),
    )
    first, status = edfm_shale_grid(
        grid, planes, rock, fracture_list=[0]
    )
    assert first.n_fracture == 1
    assert status[0].metadata["fracgrid"] == 0
    assert not status[1].metadata.get("processed", False)

    second, status = edfm_shale_grid(
        first, status, rock, fracture_list=[1]
    )
    assert second.n_fracture == 2
    assert np.array_equal(second.fracture_plane, [0, 1])
    assert [plane.metadata["fracgrid"] for plane in status] == [0, 1]

    unchanged, repeated_status = edfm_shale_grid(
        second, status, rock, fracture_list=[0, 1]
    )
    assert unchanged is second
    assert [plane.metadata["fracgrid"] for plane in repeated_status] == [0, 1]


def test_edfm_shale_grid_clips_general_convex_polygon_conservatively():
    grid = compute_geometry(cart_grid([2, 2, 1], [2.0, 2.0, 1.0]))
    triangle = ShaleFracturePlane(
        points=np.array([
            [.2, .2, .5], [1.8, .2, .5], [.2, 1.8, .5],
        ]), aperture=.01, permeability=1e-10, porosity=.5,
    )
    rock = ShaleRock(
        perm=np.full((grid.cells.num, 1), 1e-15),
        poro=np.full(grid.cells.num, .1),
        is_matrix=np.ones(grid.cells.num, dtype=bool),
    )
    global_grid, _ = edfm_shale_grid(grid, [triangle], rock)
    expected_area = 0.5 * 1.6 * 1.6
    assert global_grid.n_fracture == 3
    assert np.sum(global_grid.fracture_volumes) == pytest.approx(
        expected_area * .01
    )
    weighted_centroid = np.average(
        global_grid.fracture_centroids,
        axis=0,
        weights=global_grid.fracture_volumes,
    )
    assert np.allclose(weighted_centroid, [2.2 / 3, 2.2 / 3, .5])
    assert np.sum(global_grid.nnc.area) == pytest.approx(expected_area)
    topology = global_grid.grid
    assert topology is not None
    topology.check()
    assert topology.cells.num == grid.cells.num + global_grid.n_fracture
    assert global_grid.fracture_cell_map.tolist() == [4, 5, 6]
    assert global_grid.fracture_face_map.size > 0
    assert global_grid.fracture_node_map.size > 0
    assert np.allclose(topology.nodes.coords[global_grid.fracture_node_map, 2], .5)
    fracture_neighbors = topology.faces.neighbors[global_grid.fracture_face_map]
    # Internal fracture edges are deduplicated and shared by two fracture cells.
    assert np.count_nonzero(np.all(fracture_neighbors >= 0, axis=1)) >= 2
    assert np.allclose(
        topology.cells.volumes[global_grid.fracture_cell_map],
        global_grid.fracture_volumes,
    )


def test_edfm_shale_grid_rejects_nonconvex_polygon_instead_of_overfilling():
    grid = compute_geometry(cart_grid([1, 1, 1]))
    nonconvex = ShaleFracturePlane(
        points=np.array([
            [.1, .1, .5], [.9, .1, .5], [.5, .5, .5],
            [.9, .9, .5], [.1, .9, .5],
        ]), aperture=.01, permeability=1e-10, porosity=.5,
    )
    rock = ShaleRock(
        perm=np.array([[1e-15]]), poro=np.array([.1]),
        is_matrix=np.array([True]),
    )
    with pytest.raises(ValueError, match="convex"):
        edfm_shale_grid(grid, [nonconvex], rock)


def test_edfm_shale_grid_uses_exact_oblique_plane_cell_section():
    grid = compute_geometry(cart_grid([1, 1, 1]))
    polygon = ShaleFracturePlane(
        points=np.array([
            [0, .5, 1], [0, 1, .5], [.5, 1, 0],
            [1, .5, 0], [1, 0, .5], [.5, 0, 1],
        ]), aperture=.01, permeability=1e-10, porosity=.5,
    )
    rock = ShaleRock(
        perm=np.array([[1e-15]]), poro=np.array([.1]),
        is_matrix=np.array([True]),
    )
    global_grid, _ = edfm_shale_grid(grid, [polygon], rock)
    expected_area = 3 * np.sqrt(3) / 4
    assert global_grid.n_fracture == 1
    assert global_grid.fracture_volumes[0] == pytest.approx(expected_area * .01)
    assert np.allclose(global_grid.fracture_centroids[0], [.5, .5, .5])
    assert global_grid.nnc.area[0] == pytest.approx(expected_area)


def test_edfm_shale_grid_clips_against_actual_tetrahedral_cell_edges():
    points = np.array([
        [0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.],
    ])
    grid = compute_geometry(tetrahedral_grid(points, [[0, 1, 2, 3]]))
    plane = ShaleFracturePlane(
        points=np.array([
            [-.1, -.1, .25], [1.1, -.1, .25],
            [1.1, 1.1, .25], [-.1, 1.1, .25],
        ]), aperture=.01, permeability=1e-10, porosity=.5,
    )
    rock = ShaleRock(
        perm=np.array([[1e-15]]), poro=np.array([.1]),
        is_matrix=np.array([True]),
    )
    global_grid, _ = edfm_shale_grid(grid, [plane], rock)
    expected_area = .5 * .75**2
    assert global_grid.n_fracture == 1
    assert global_grid.nnc.area[0] == pytest.approx(expected_area)
    assert global_grid.fracture_volumes[0] == pytest.approx(expected_area * .01)
    assert np.allclose(global_grid.fracture_centroids[0], [.25, .25, .25])
    global_grid.grid.check()


def test_setup_pedfm_operators_combines_faces_nncs_and_conserves_flux():
    grid = _two_cell_grid()
    nnc = ShaleNNC(
        cells=[[1, 2]],
        transmissibility=[3.0],
        connection_type=("fracmat",),
        area=[1.0],
        normal=[[1.0, 0.0, 0.0]],
        plane_point=[[1.0, 0.5, 0.5]],
    )
    rock = ShaleRock(
        perm=np.ones((3, 1)),
        poro=np.array([0.2, 0.2, 0.5]),
        is_matrix=np.array([True, True, False]),
        shale_mechanisms={"sorption": True},
    )
    operators = setup_pedfm_operators_tpfa(
        grid,
        rock,
        np.full(grid.faces.num, 2.0),
        nnc,
        projected_matrix_neighbors=[[0, 1]],
        projected_areas=[0.25],
    )
    assert np.allclose(operators.T, [1.5, 3.0])
    divergence = operators.div([4.0, 5.0])
    assert np.allclose(divergence, [-4.0, -1.0, 5.0])
    assert np.sum(divergence) == pytest.approx(0.0)
    assert np.array_equal(operators.is_sorbed, [True, True, False])
    assert np.allclose(operators.grain_volume, [0.8, 0.8, 0.0])


def test_setup_pedfm_operators_uses_positive_global_fracture_volumes():
    grid = _two_cell_grid()
    nnc = ShaleNNC(
        cells=[[1, 2]], transmissibility=[3.0],
        connection_type=("fracmat",), area=[1.0],
        normal=[[1.0, 0.0, 0.0]], plane_point=[[1.0, 0.5, 0.5]],
    )
    rock = ShaleRock(
        perm=np.ones((3, 1)), poro=np.array([.2, .2, .5]),
        is_matrix=np.array([True, True, False]),
    )
    operators = setup_pedfm_operators_tpfa(
        grid, rock, np.full(grid.faces.num, 2.0), nnc,
        cell_volumes=[1.0, 1.0, .02],
    )
    assert np.allclose(operators.pv, [.2, .2, .01])
    with pytest.raises(ValueError, match="one value per global cell"):
        setup_pedfm_operators_tpfa(
            grid, rock, np.full(grid.faces.num, 2.0), nnc,
            cell_volumes=[1.0, 1.0],
        )


def test_native_shale_edfm_operators_cover_full_simulation_contract():
    grid = _two_cell_grid()
    nnc = ShaleNNC(
        cells=[[1, 2]], transmissibility=[3.0],
        connection_type=("fracmat",), area=[1.0],
        normal=[[1.0, 0.0, 0.0]], plane_point=[[1.0, 0.5, 0.5]],
    )
    rock = ShaleRock(
        perm=np.ones((3, 1)), poro=np.array([0.2, 0.2, 0.5]),
        is_matrix=np.array([True, True, False]),
        shale_mechanisms={"sorption": True},
    )
    operators = setup_shale_edfm_operators_tpfa(
        grid, rock, np.full(grid.faces.num, 2.0), nnc,
        cell_volumes=[1.0, 1.0, 0.02],
    )
    assert np.allclose(operators.T, [2.0, 3.0])
    assert np.allclose(operators.pv, [0.2, 0.2, 0.01])
    assert np.allclose(operators.grain_volume, [0.8, 0.8, 0.0])
    assert np.allclose(operators.grad([1.0, 3.0, 7.0]), [2.0, 4.0])
    assert np.allclose(operators.avg([1.0, 3.0, 7.0]), [2.0, 5.0])
    assert np.allclose(
        operators.upw([True, False], [1.0, 3.0, 7.0]), [1.0, 7.0]
    )
    assert np.allclose(operators.Div([4.0, 5.0]), [-4.0, -1.0, 5.0])


def test_shale_connection_operators_preserve_adi_jacobians():
    grid = _two_cell_grid()
    nnc = ShaleNNC(
        cells=[[1, 2]], transmissibility=[3.0],
        connection_type=("fracmat",), area=[1.0],
        normal=[[1.0, 0.0, 0.0]], plane_point=[[1.0, 0.5, 0.5]],
    )
    rock = ShaleRock(
        perm=np.ones((3, 1)), poro=np.full(3, 0.2),
        is_matrix=np.array([True, True, False]),
    )
    operators = setup_shale_edfm_operators_tpfa(
        grid, rock, np.ones(grid.faces.num), nnc,
        cell_volumes=np.ones(3),
    )
    (pressure,) = init_variables_adi(np.array([1.0, 3.0, 7.0]))
    gradient = operators.grad(pressure)
    upwind = operators.upw(np.array([True, False]), pressure)
    divergence = operators.div(gradient)
    assert np.allclose(value(gradient), [2.0, 4.0])
    assert gradient.jac[0].shape == (2, 3)
    assert upwind.jac[0].shape == (2, 3)
    assert divergence.jac[0].shape == (3, 3)


def test_projected_matrix_fracture_nncs_select_three_interior_directions():
    grid = compute_geometry(cart_grid([3, 3, 3], [3.0, 3.0, 3.0]))
    fracture = PlanarFracture(
        center=np.array([1.5, 1.5, 1.5]),
        normal=np.array([1.0, 0.0, 0.0]),
        axis1=np.array([0.0, 1.0, 0.0]),
        half_length1=0.35,
        half_length2=0.35,
    )
    projection = project_planar_fracture_to_grid(grid, fracture, aperture=0.01)
    assert projection.matrix_cells.size == 1
    edfm = build_edfm_3d(
        grid,
        [fracture],
        aperture=0.01,
        perm_matrix=np.full(grid.cells.num, 1.0e-15),
        perm_frac=1.0e-10,
        poro_matrix=np.full(grid.cells.num, 0.1),
    )
    projected = projected_matrix_fracture_nncs_3d(
        edfm, [fracture], [projection]
    )
    assert projected.MFcells.shape == (3, 2)
    assert projected.MMneighs.shape == (3, 2)
    assert np.array_equal(np.sort(projected.direction), [1, 2, 3])
    assert np.count_nonzero(projected.area > 0) == 1
    assert np.count_nonzero(projected.T > 0) == 1
    host = int(projection.matrix_cells[0])
    assert all(host in pair for pair in projected.MMneighs)


def test_projected_nncs_drop_domain_boundary_directions():
    grid = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))
    fracture = PlanarFracture(
        center=np.array([0.5, 0.5, 0.5]),
        normal=np.array([1.0, 0.0, 0.0]),
        axis1=np.array([0.0, 1.0, 0.0]),
        half_length1=0.25,
        half_length2=0.25,
    )
    projection = project_planar_fracture_to_grid(grid, fracture, aperture=0.01)
    edfm = build_edfm_3d(
        grid,
        [fracture],
        aperture=0.01,
        perm_matrix=np.full(grid.cells.num, 1.0e-15),
        perm_frac=1.0e-10,
        poro_matrix=np.full(grid.cells.num, 0.1),
    )
    projected = projected_matrix_fracture_nncs_3d(edfm, [fracture], [projection])
    # The deterministic tie branch selects the negative faces, all boundaries.
    assert projected.MFcells.shape == (0, 2)


def test_projected_nncs_feed_pedfm_operator_without_manual_repacking():
    grid = compute_geometry(cart_grid([3, 3, 3], [3.0, 3.0, 3.0]))
    fracture = PlanarFracture(
        center=np.array([1.5, 1.5, 1.5]),
        normal=np.array([1.0, 0.0, 0.0]),
        axis1=np.array([0.0, 1.0, 0.0]),
        half_length1=0.35,
        half_length2=0.35,
    )
    projection = project_planar_fracture_to_grid(grid, fracture, aperture=0.01)
    edfm = build_edfm_3d(
        grid,
        [fracture],
        aperture=0.01,
        perm_matrix=np.full(grid.cells.num, 1.0e-15),
        perm_frac=1.0e-10,
        poro_matrix=np.full(grid.cells.num, 0.1),
    )
    normal_nnc = fracture_matrix_shale_nncs_3d(edfm, [fracture], [projection])
    projected = projected_matrix_fracture_nncs_3d(edfm, [fracture], [projection])
    rock = assemble_shale_global_grid(edfm).rock
    operators = setup_pedfm_operators_tpfa(
        grid,
        rock,
        np.ones(grid.faces.num),
        normal_nnc,
        projected=projected,
    )
    internal_count = np.count_nonzero(np.all(grid.faces.neighbors >= 0, axis=1))
    assert operators.N.shape[0] == (
        internal_count + normal_nnc.cells.shape[0] + projected.MFcells.shape[0]
    )
    assert operators.divergence_matrix.shape[0] == edfm.n_total
    flux = np.arange(operators.N.shape[0], dtype=float)
    assert np.sum(operators.div(flux)) == pytest.approx(0.0)


def test_projected_nncs_use_exact_partial_overlap_union_area():
    grid = compute_geometry(cart_grid([3, 3, 3], [3.0, 3.0, 3.0]))
    fractures = [
        PlanarFracture(
            center=np.array([1.5, y, 1.5]), normal=np.array([1., 0., 0.]),
            axis1=np.array([0., 1., 0.]), half_length1=half,
            half_length2=.3,
        )
        for y, half in [(1.4, .3), (1.65, .25)]
    ]
    projections = [
        project_planar_fracture_to_grid(grid, fracture, aperture=.01)
        for fracture in fractures
    ]
    edfm = build_edfm_3d(
        grid, fractures, aperture=.01,
        perm_matrix=np.full(grid.cells.num, 1e-15), perm_frac=1e-10,
        poro_matrix=np.full(grid.cells.num, .1),
    )
    result = projected_matrix_fracture_nncs_3d(edfm, fractures, projections)
    x_projection = result.area[result.direction == 1]
    assert x_projection.size == 1
    # Areas .36 and .30 overlap over .30*.60=.18.
    assert x_projection[0] == pytest.approx(.36 + .30 - .18)


def test_internal_convex_polygon_union_handles_overlap_containment_and_disjoint():
    from pymrst.modules.shale.pedfm_nnc.projected_matrix_fracture_nncs_3d import (
        _convex_polygon_union_area,
    )
    square = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    overlap = square + [0.5, 0]
    contained = square * .5 + .25
    disjoint = square + [2., 0]
    assert _convex_polygon_union_area([square, overlap]) == pytest.approx(1.5)
    assert _convex_polygon_union_area([square, contained]) == pytest.approx(1.0)
    assert _convex_polygon_union_area([square, disjoint]) == pytest.approx(2.0)


def test_projected_nncs_support_affine_skew_hexahedral_cells():
    grid = cart_grid([3, 3, 3], [3.0, 3.0, 3.0])
    logical = grid.nodes.coords.copy()
    grid.nodes.coords[:, 0] = logical[:, 0] + .25 * logical[:, 1] + .1 * logical[:, 2]
    grid = compute_geometry(grid)
    normal = np.array([1., -.25, -.1])
    normal /= np.linalg.norm(normal)
    axis1 = np.array([.25, 1., 0.])
    center = np.array([1.5 + .25 * 1.5 + .1 * 1.5, 1.5, 1.5])
    fracture = PlanarFracture(
        center=center, normal=normal, axis1=axis1,
        half_length1=.3, half_length2=.3,
    )
    projection = project_planar_fracture_to_grid(grid, fracture, aperture=.01)
    edfm = build_edfm_3d(
        grid, [fracture], aperture=.01,
        perm_matrix=np.full(grid.cells.num, 1e-15), perm_frac=1e-10,
        poro_matrix=np.full(grid.cells.num, .1),
    )
    result = projected_matrix_fracture_nncs_3d(edfm, [fracture], [projection])
    assert result.MFcells.shape == (3, 2)
    assert np.array_equal(np.sort(result.direction), [1, 2, 3])
    assert np.all(result.area >= 0)


def test_fracture_fracture_nncs_detect_finite_cross_plane_intersection():
    grid = compute_geometry(cart_grid([3, 3, 3], [3.0, 3.0, 3.0]))
    fractures = [
        PlanarFracture(
            center=np.array([1.5, 1.5, 1.5]), normal=np.array([1.0, 0.0, 0.0]),
            axis1=np.array([0.0, 1.0, 0.0]), half_length1=0.8, half_length2=0.8,
        ),
        PlanarFracture(
            center=np.array([1.5, 1.5, 1.5]), normal=np.array([0.0, 1.0, 0.0]),
            axis1=np.array([1.0, 0.0, 0.0]), half_length1=0.8, half_length2=0.8,
        ),
    ]
    projections = [
        project_planar_fracture_to_grid(grid, fracture, aperture=0.01)
        for fracture in fractures
    ]
    edfm = build_edfm_3d(
        grid,
        fractures,
        aperture=0.01,
        perm_matrix=np.full(grid.cells.num, 1.0e-15),
        perm_frac=1.0e-10,
        poro_matrix=np.full(grid.cells.num, 0.1),
    )
    result = fracture_fracture_shale_nncs_3d(
        edfm, fractures, projections, aperture=0.01
    )
    # The 1.6 m intersection crosses three fracture cells on each plane and
    # is therefore represented by three conservative cell-pair NNCs.
    assert result.cells.shape == (3, 2)
    assert result.type == ("fracfrac",) * 3
    assert np.all(result.cells[:, 0] < result.cells[:, 1])
    assert np.all(result.T > 0)
    assert np.unique(result.cells, axis=0).shape[0] == 3

    from pymrst.modules.shale.hfm_fns.fracture_fracture_shale_nncs_3d import (
        _cell_line_intervals,
    )
    segment = _rectangle_intersection_segment_for_test(fractures[0], fractures[1])
    start, end = segment
    direction = end - start
    length = np.linalg.norm(direction)
    direction /= length
    intervals = _cell_line_intervals(
        start, direction, length, projections[0], fractures[0], 1e-10
    )
    assert sum(upper - lower for _, lower, upper in intervals) == pytest.approx(1.6)

    separated = list(fractures)
    separated[1] = PlanarFracture(
        center=np.array([1.5, 1.5, 4.0]), normal=np.array([0.0, 1.0, 0.0]),
        axis1=np.array([1.0, 0.0, 0.0]), half_length1=0.8, half_length2=0.2,
    )
    assert _rectangle_intersection_segment_for_test(fractures[0], separated[1]) is None


def _rectangle_intersection_segment_for_test(first, second):
    # Exercise the public result through an empty/non-intersecting grid setup is
    # unnecessarily expensive; this mirrors the finite extent assertion by
    # importing the owner-local geometry helper.
    from pymrst.modules.shale.hfm_fns.fracture_fracture_shale_nncs_3d import (
        _rectangle_intersection_segment,
    )
    return _rectangle_intersection_segment(first, second, 1e-10)


def test_fracture_fracture_nnc_uses_two_sided_resistance_for_mixed_properties():
    grid = compute_geometry(cart_grid([1, 1, 1]))
    fractures = [
        PlanarFracture(
            center=np.array([.5, .5, .5]), normal=np.array([1., 0., 0.]),
            axis1=np.array([0., 1., 0.]), half_length1=.3, half_length2=.3,
        ),
        PlanarFracture(
            center=np.array([.5, .5, .5]), normal=np.array([0., 1., 0.]),
            axis1=np.array([1., 0., 0.]), half_length1=.3, half_length2=.3,
        ),
    ]
    apertures = np.array([.01, .02])
    projections = [
        project_planar_fracture_to_grid(grid, fracture, aperture=apertures[i])
        for i, fracture in enumerate(fractures)
    ]
    edfm = build_edfm_3d(
        grid, fractures, aperture=apertures,
        perm_matrix=np.array([1e-15]), perm_frac=[1e-10, 3e-10],
        poro_matrix=np.array([.1]), poro_frac=[.4, .7],
    )
    result = fracture_fracture_shale_nncs_3d(
        edfm, fractures, projections, aperture=apertures
    )
    assert result.cells.shape == (1, 2)
    midpoint = np.array([.5, .5, .5])
    d1 = max(np.linalg.norm(projections[0].centroids[0] - midpoint), .005)
    d2 = max(np.linalg.norm(projections[1].centroids[0] - midpoint), .01)
    expected = 1.0 / (
        d1 / (1e-10 * .01 * .6) + d2 / (3e-10 * .02 * .6)
    )
    assert result.T[0] == pytest.approx(expected)
