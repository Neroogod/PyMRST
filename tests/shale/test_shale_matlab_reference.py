from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.shale import (
    create_multistage_hfs,
    compute_wi_shale,
    add_well_shale,
    assemble_shale_global_grid,
    gangi_fn,
    make_shale_rock,
    points_to_cell_indices,
    projected_matrix_fracture_nncs_3d,
    well_fracture_nncs_3d,
)
from pymrst.modules.hfm import (
    PlanarFracture,
    build_edfm_3d,
    project_planar_fracture_to_grid,
)


FIXTURE = Path(__file__).parents[2] / "tests" / "fixtures" / "shale" / "foundation.mat"
PEDFM_FIXTURE = FIXTURE.with_name("pedfm_projection.mat")


@pytest.mark.skipif(not FIXTURE.exists(), reason="generate fixture with MATLAB script")
def test_shale_foundation_matches_direct_mrst_fixture():
    ref = loadmat(FIXTURE, simplify_cells=True)
    grid = compute_geometry(cart_grid([2, 2, 1], [3.0, 5.0, 4.0]))
    points = np.array([[0.25, 0.25, 1], [2, 0.25, 1], [0.25, 3, 1], [2, 3, 1]])
    assert np.array_equal(
        points_to_cell_indices(points, grid),
        np.asarray(ref["point_cell_ids_one_based"]).reshape(-1) - 1,
    )
    planes, centers = create_multistage_hfs(
        num_stages=2,
        frac_spacing=10,
        num_fracs_per_stage=2,
        frac_half_length=[4, 5],
        frac_height=[6, 8],
        cluster_spacing=2,
        frac_centroid1=[10, 20, 15],
    )
    assert np.allclose(centers, np.asarray(ref["frac_centroids"]).reshape(-1))
    matlab_points = np.asarray(ref["fracture_points"])
    assert np.allclose(np.stack([plane.points for plane in planes], axis=2), matlab_points)
    assert np.allclose(
        gangi_fn(np.array([0.0, 2.0, 4.0]), 10, 0.5, 20, 2, 4, 2),
        np.asarray(ref["gangi"]).reshape(-1),
    )
    rock = make_shale_rock(grid, [[10, 5, 1]], [0.2, 0.1, 0.3, 0.4], ntg=0.8)
    assert np.allclose(rock.perm, ref["rock_perm"])
    assert np.allclose(rock.poro, np.asarray(ref["rock_poro"]).reshape(-1))
    assert np.allclose(rock.ntg, np.asarray(ref["rock_ntg"]).reshape(-1))
    assert np.array_equal(rock.isMatrix, np.asarray(ref["rock_is_matrix"]).reshape(-1).astype(bool))


@pytest.mark.skipif(not PEDFM_FIXTURE.exists(), reason="generate fixture with MATLAB script")
def test_projected_nncs_match_direct_mrst_fixture():
    ref = loadmat(PEDFM_FIXTURE, simplify_cells=True)
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
    result = projected_matrix_fracture_nncs_3d(edfm, [fracture], [projection])
    assert np.array_equal(result.MFcells, ref["projected_mf_one_based"] - 1)
    assert np.array_equal(result.MMneighs, ref["projected_mm_one_based"] - 1)
    assert np.array_equal(result.direction, ref["projected_direction"])
    assert np.allclose(result.area, ref["projected_area"], rtol=1e-12, atol=1e-14)
    assert np.allclose(result.T, ref["projected_T"], rtol=1e-10, atol=1e-28)
    wi = compute_wi_shale(
        1.0e-10,
        0.01,
        0.05,
        cell_side_lengths=[0.7, 0.7],
    )
    assert wi.item() == pytest.approx(float(ref["fracture_wi"]), rel=1e-12)
    intersections = well_fracture_nncs_3d(
        [np.array([[0.0, 1.5, 1.5], [3.0, 1.5, 1.5]])],
        [fracture],
        [projection],
        matrix_cell_count=grid.cells.num,
    )
    assert np.allclose(intersections.points, np.atleast_2d(ref["well_intersection_points"]))
    assert np.array_equal(
        intersections.fracture_index,
        np.asarray(ref["well_intersection_plane_one_based"]).reshape(-1) - 1,
    )
    assert np.array_equal(
        intersections.fracture_cells,
        np.asarray(ref["well_intersection_cell_one_based"]).reshape(-1) - 1,
    )
    global_grid = assemble_shale_global_grid(edfm)
    well = add_well_shale(
        [], global_grid, [[global_grid.n_matrix, 0]],
        fracture_aperture=0.01,
        representative_radius=float(np.asarray(ref["well_rR"]).reshape(-1)[0] ** 2 / 0.05),
        radius=0.05, well_type="rate", val=-1, name="P1",
        compi=[0.9, 0.1], sign=1, ref_depth=0, vfp_index=7, c=[4, 5],
    )[0]
    assert np.array_equal(well.cells, np.asarray(ref["well_cells_one_based"]).reshape(-1) - 1)
    assert np.allclose(well.WI, np.asarray(ref["well_WI"]).reshape(-1), rtol=1e-12)
    assert np.allclose(well.rR, np.asarray(ref["well_rR"]).reshape(-1), rtol=1e-12)
    assert np.allclose(well.dZ, np.asarray(ref["well_dZ"]).reshape(-1), rtol=1e-12)
    assert well.sign == int(ref["well_sign"]) == -1
    assert well.vfp_index == int(ref["well_vfp_index"]) == 7
    assert np.array_equal(well.c, np.asarray(ref["well_c"]).reshape(-1))
