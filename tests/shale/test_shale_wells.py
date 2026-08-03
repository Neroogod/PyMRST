import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.hfm import (
    PlanarFracture,
    build_edfm_3d,
    project_planar_fracture_to_grid,
)
from pymrst.modules.shale import (
    add_well_shale,
    assemble_shale_global_grid,
    compute_wi_shale,
    well_fracture_nncs_3d,
)


def _global_grid_with_one_fracture():
    grid = compute_geometry(cart_grid([3, 3, 3], [3.0, 3.0, 3.0]))
    fracture = PlanarFracture(
        center=np.array([1.5, 1.5, 1.5]),
        normal=np.array([1.0, 0.0, 0.0]),
        axis1=np.array([0.0, 1.0, 0.0]),
        half_length1=0.35,
        half_length2=0.35,
    )
    edfm = build_edfm_3d(
        grid,
        [fracture],
        aperture=0.01,
        perm_matrix=np.full(grid.cells.num, 1e-15),
        perm_frac=1e-10,
        poro_matrix=np.full(grid.cells.num, 0.1),
    )
    return assemble_shale_global_grid(edfm)


def test_compute_wi_shale_matches_mrst_rectangular_formula():
    permeability = np.array([1.0e-10, 2.0e-10])
    aperture = np.array([0.01, 0.02])
    sides = np.array([[1.0, 2.0], [2.0, 2.0]])
    radius = 0.05
    skin = np.array([0.0, 0.5])
    actual = compute_wi_shale(
        permeability,
        aperture,
        radius,
        cell_side_lengths=sides,
        skin=skin,
    )
    representative = 0.14 * np.sqrt(np.sum(sides**2, axis=1))
    expected = 2 * np.pi * permeability * aperture / (
        np.log(representative / radius) + skin
    )
    assert np.allclose(actual, expected)


def test_compute_wi_shale_disables_natural_fracture_completions():
    result = compute_wi_shale(
        [1e-10, 1e-10],
        0.01,
        0.05,
        representative_radius=[0.2, 0.2],
        hydraulic_completion=[True, False],
    )
    assert result[0] > 0
    assert result[1] == 0
    with pytest.raises(ValueError, match="denominator"):
        compute_wi_shale(1e-10, 0.01, 0.2, representative_radius=0.1)


def test_well_fracture_intersection_maps_point_to_global_fracture_cell():
    grid = compute_geometry(cart_grid([3, 3, 3], [3.0, 3.0, 3.0]))
    fracture = PlanarFracture(
        center=np.array([1.5, 1.5, 1.5]),
        normal=np.array([1.0, 0.0, 0.0]),
        axis1=np.array([0.0, 1.0, 0.0]),
        half_length1=0.4,
        half_length2=0.4,
    )
    projection = project_planar_fracture_to_grid(grid, fracture, aperture=0.01)
    result = well_fracture_nncs_3d(
        [np.array([[0.0, 1.5, 1.5], [3.0, 1.5, 1.5]])],
        [fracture],
        [projection],
        matrix_cell_count=grid.cells.num,
    )
    assert result.points.shape == (1, 3)
    assert np.allclose(result.points[0], [1.5, 1.5, 1.5])
    assert result.fracture_index[0] == 0
    assert result.fracture_cells[0] >= grid.cells.num
    local = result.fracture_cells[0] - grid.cells.num
    assert result.matrix_cells[0] == projection.matrix_cells[local]


def test_well_fracture_intersection_rejects_outside_polygon_and_deduplicates_vertex():
    grid = compute_geometry(cart_grid([3, 3, 3], [3.0, 3.0, 3.0]))
    fracture = PlanarFracture(
        center=np.array([1.5, 1.5, 1.5]),
        normal=np.array([1.0, 0.0, 0.0]),
        axis1=np.array([0.0, 1.0, 0.0]),
        half_length1=0.4,
        half_length2=0.4,
    )
    projection = project_planar_fracture_to_grid(grid, fracture, aperture=0.01)
    result = well_fracture_nncs_3d(
        [
            np.array([[0.0, 2.5, 2.5], [3.0, 2.5, 2.5]]),
            np.array([[0.0, 1.5, 1.5], [1.5, 1.5, 1.5], [3.0, 1.5, 1.5]]),
        ],
        [fracture],
        [projection],
        matrix_cell_count=grid.cells.num,
    )
    assert result.points.shape == (1, 3)
    assert result.well_index[0] == 1


def test_add_well_shale_builds_normal_pymrst_well_and_disables_nf_completion():
    grid = compute_geometry(cart_grid([3, 3, 3], [3.0, 3.0, 3.0]))
    fractures = [
        PlanarFracture(
            center=np.array([1.5, 1.5, 1.5]), normal=np.array([1.0, 0.0, 0.0]),
            axis1=np.array([0.0, 1.0, 0.0]), half_length1=0.35, half_length2=0.35,
        ),
        PlanarFracture(
            center=np.array([2.5, 1.5, 1.5]), normal=np.array([1.0, 0.0, 0.0]),
            axis1=np.array([0.0, 1.0, 0.0]), half_length1=0.35, half_length2=0.35,
        ),
    ]
    edfm = build_edfm_3d(
        grid, fractures, aperture=0.01,
        perm_matrix=np.full(grid.cells.num, 1e-15), perm_frac=1e-10,
        poro_matrix=np.full(grid.cells.num, 0.1),
    )
    global_grid = assemble_shale_global_grid(edfm)
    first_plane_cell = global_grid.n_matrix
    second_plane_cell = global_grid.n_matrix + np.count_nonzero(edfm.frac_line == 0)
    wells = add_well_shale(
        None,
        global_grid,
        [[first_plane_cell, 0], [second_plane_cell, 1]],
        fracture_aperture=[0.01, 0.02],
        representative_radius=[0.2, 0.25],
        radius=0.05,
        hydraulic_plane_count=1,
        well_type="rate",
        val=-1.0,
        name="P1",
        components=[0.9, 0.1],
    )
    assert len(wells) == 1
    well = wells[0]
    assert well.name == "P1"
    assert well.sign == -1
    assert well.WI[0] > 0
    assert well.WI[1] == 0
    assert np.array_equal(well.cp["fracture_plane"], [0, 1])
    assert np.allclose(well.rR, np.sqrt(np.array([0.2, 0.25]) * 0.05))


def test_add_well_shale_accepts_matrix_cell_vector_without_fracture_options():
    global_grid = _global_grid_with_one_fracture()
    well = add_well_shale(
        [], global_grid, [0, 1], radius=0.05, direction="xy",
        well_type="rate", val=1.0, sign=-1, vfp_index=7,
        c={"legacy": 3},
    )[0]
    assert np.array_equal(well.cells, [0, 1])
    assert np.array_equal(well.cp["fracture_plane"], [-1, -1])
    assert np.all(well.WI > 0)
    assert well.dir == "xy"
    assert well.sign == 1
    assert well.vfp_index == 7
    assert well.c == {"legacy": 3}
    assert np.all(well.rR > 0)


def test_add_well_shale_accepts_logical_mask_and_mixed_completions():
    global_grid = _global_grid_with_one_fracture()
    mask = np.zeros(global_grid.n_cells, dtype=bool)
    mask[0] = True
    mask[global_grid.n_matrix] = True
    well = add_well_shale(
        None,
        global_grid,
        mask,
        fracture_aperture=0.01,
        representative_radius=[0.2, 0.25],
        radius=0.05,
        calculate_representative_radius=False,
    )[0]
    assert np.array_equal(well.cells, [0, global_grid.n_matrix])
    assert np.array_equal(well.cp["fracture_plane"], [-1, 0])
    assert np.all(well.WI > 0)
    assert well.rR is None


def test_add_well_shale_recomputes_only_negative_supplied_indices():
    global_grid = _global_grid_with_one_fracture()
    well = add_well_shale(
        None,
        global_grid,
        [0, 1],
        wi=[123.0, -1.0],
        cell_dims_override=np.ones((global_grid.n_matrix, 3)),
    )[0]
    assert well.WI[0] == 123.0
    assert well.WI[1] > 0


def test_add_well_shale_requires_geometry_for_fracture_representative_radius():
    global_grid = _global_grid_with_one_fracture()
    with pytest.raises(ValueError, match="fracture_cell_side_lengths"):
        add_well_shale(
            None,
            global_grid,
            [global_grid.n_matrix],
            fracture_aperture=0.01,
            radius=0.05,
        )
