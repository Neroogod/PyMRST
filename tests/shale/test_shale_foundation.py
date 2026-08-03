import numpy as np
import pytest

from pymrst.gridprocessing import cart_grid, tensor_grid
from pymrst.modules.shale import (
    ShaleRock,
    SorbedCompositionalFluid,
    available_shale_datasets,
    create_multistage_hfs,
    gangi_fn,
    load_cmg_timesteps,
    load_shale_dataset,
    make_shale_rock,
    mesh_hf_system,
    points_to_cell_indices,
    process_stochastic_fractures,
)


def test_make_shale_rock_preserves_mrst_fields_and_tensor_layout():
    grid = cart_grid([2, 1, 1])
    rock = make_shale_rock(
        grid,
        [[10.0, 5.0, 1.0]],
        [0.2, 0.1],
        ntg=0.8,
        is_matrix=[True, False],
        shale_mechanisms={"Gangi": {"m": 0.5}},
    )
    assert isinstance(rock, ShaleRock)
    assert rock.perm.shape == (2, 3)
    assert np.allclose(rock.ntg, 0.8)
    assert np.array_equal(rock.isMatrix, [True, False])
    assert "Gangi" in rock.shaleMechanisms


def test_gangi_matches_mrst_expression_and_accepts_arrays():
    pressure = np.array([0.0, 2.0, 4.0])
    actual = gangi_fn(pressure, pc=10.0, alpha=0.5, pmax=20.0, m=2.0, k0=4.0, matrix_perm=2.0)
    expected = 2.0 * (1.0 - ((10.0 - 0.5 * pressure) / 20.0) ** 2.0) ** 3.0
    assert np.allclose(actual, expected)
    with pytest.raises(ValueError, match="pmax"):
        gangi_fn(1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0)


def test_points_to_cell_indices_uses_mrst_column_major_numbering():
    grid = tensor_grid(np.array([0.0, 1.0, 3.0]), np.array([0.0, 2.0, 5.0]), np.array([0.0, 4.0]))
    points = np.array([[0.25, 0.25, 1.0], [2.0, 0.25, 1.0], [0.25, 3.0, 1.0], [2.0, 3.0, 1.0]])
    assert np.array_equal(points_to_cell_indices(points, grid), [0, 1, 2, 3])
    with pytest.raises(ValueError, match="outside"):
        points_to_cell_indices([[4.0, 1.0, 1.0]], grid)


def test_create_multistage_hfs_matches_reference_centroid_progression():
    planes, centers = create_multistage_hfs(
        num_stages=2,
        frac_spacing=10.0,
        num_fracs_per_stage=2,
        frac_half_length=[4.0, 5.0],
        frac_height=[6.0, 8.0],
        cluster_spacing=2.0,
        frac_centroid1=[10.0, 20.0, 15.0],
    )
    assert np.array_equal(centers, [10.0, 12.0, 20.0, 22.0])
    assert len(planes) == 4
    assert np.allclose(planes[0].points[:, 0], 10.0)
    assert np.ptp(planes[0].points[:, 1]) == 8.0
    assert np.ptp(planes[0].points[:, 2]) == 6.0


def test_mesh_hf_system_places_every_fracture_on_an_x_face():
    grid = mesh_hf_system(
        [30.0, 20.0, 10.0],
        [10.0, 12.0, 20.0, 22.0],
        num_stages=2,
        num_fracs_per_stage=2,
        nx_refine=2,
        nx_refine_small=2,
        ny=3,
        nz=3,
    )
    xcoords = np.unique(grid.nodes.coords[:, 0])
    assert all(np.any(np.isclose(xcoords, x)) for x in [10.0, 12.0, 20.0, 22.0])
    assert grid.cells.num == (xcoords.size - 1) * 4


def test_process_stochastic_fractures_filters_and_classifies():
    closing = np.array([[0, 0, 0], [2, 0, 0], [2, 1, 0], [0, 1, 0], [0, 0, 0]], dtype=float)
    warped = closing.copy()
    warped[2, 2] = 0.5
    nan_polygon = closing.copy()
    nan_polygon[1, 0] = np.nan
    planar, nonplanar, areas = process_stochastic_fractures([closing, warped, nan_polygon])
    assert len(planar) == 1
    assert len(nonplanar) == 1
    assert areas[0] == pytest.approx(2.0)
    assert areas[2] == 0.0


def test_sorbed_compositional_fluid_retains_component_isotherms():
    fluid = SorbedCompositionalFluid(
        ["Methane", "Ethane", "n-Propane"],
        [[10769611.0, 5591648.0, 5819175.0], [2.99, 4.86, 9.56]],
    )
    assert fluid.n == 3
    assert np.array_equal(fluid.langmuir_pressure, fluid.isotherm[0])
    assert np.array_equal(fluid.langmuir_density, fluid.isotherm[1])
    with pytest.raises(ValueError, match="isotherm"):
        SorbedCompositionalFluid(["Methane"], [1.0, 2.0])


def test_original_mrst_shale_assets_are_packaged_and_loadable():
    assert len(available_shale_datasets()) == 5
    fractures = load_shale_dataset("stochastic_fractures")
    assert set(fractures) == {"set1", "set2"}
    assert len(fractures["set1"]) == 350
    timesteps = load_cmg_timesteps()
    assert timesteps.ndim == 1
    assert timesteps.size > 10
    # Preserve the distributed CMG sequence exactly, including repeated and
    # slightly decreasing late report values.
    assert timesteps[0] == pytest.approx(1.0e-6)
    assert np.all(timesteps > 0)
