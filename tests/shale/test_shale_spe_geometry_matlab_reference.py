"""Direct MRST certification for SPE-201243-PA geometry owners."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.modules.shale.spe201243pa import (
    explicit_stencil,
    explicit_stencil_3_fracs,
    multistage_planar_nonplanar_hf,
    structured_grid_fracture_on_face,
)


FIXTURE = Path(__file__).parents[2] / "tests/fixtures/shale/spe_geometry.mat"


def _coordinates(grid, axis):
    return np.unique(grid.nodes.coords[:, axis])


def test_multistage_inclined_fractures_match_direct_mrst_fixture():
    ref = loadmat(FIXTURE, simplify_cells=True)
    planes, centers = multistage_planar_nonplanar_hf(
        num_stages=2, fracture_spacing=10, fractures_per_stage=3,
        fracture_half_length=[4, 5], fracture_height=[6, 8], theta=.4,
        cluster_spacing=2, heel_coordinate=[10, 20, 15],
    )
    points = np.stack([plane.points for plane in planes], axis=2)
    assert np.allclose(centers, ref["multistage_centers"])
    assert np.allclose(points, ref["multistage_points"])
    assert [plane.metadata["stage"] for plane in planes] == [0, 0, 0, 1, 1, 1]


def test_structured_fracture_face_grid_matches_direct_mrst_fixture():
    ref = loadmat(FIXTURE, simplify_cells=True)
    grid = structured_grid_fracture_on_face(
        [100, 200, 60], ref["multistage_centers"], num_stages=2,
        fractures_per_stage=3, fracture_spacing=10, nx_refine=10,
        nx_refine_small=2, ny=11, nz=11,
    )
    assert np.array_equal(grid.cartDims, ref["structured_dims"])
    for axis, name in enumerate("xyz"):
        assert np.allclose(_coordinates(grid, axis), ref[f"structured_{name}"])


def _assert_stencil(prefix, result, ref):
    grid, fracture_ids, well_ids = result
    assert np.array_equal(grid.cartDims, ref[f"{prefix}_dims"])
    for axis, name in enumerate("xyz"):
        assert np.allclose(_coordinates(grid, axis), ref[f"{prefix}_{name}"])
    expected_ids = np.asarray(ref[f"{prefix}_ids"]).reshape(fracture_ids.shape) - 1
    assert np.array_equal(fracture_ids, expected_ids)
    assert np.array_equal(well_ids, np.atleast_1d(ref[f"{prefix}_wells"]) - 1)


def test_explicit_single_fracture_stencil_matches_direct_mrst_fixture():
    ref = loadmat(FIXTURE, simplify_cells=True)
    _assert_stencil(
        "explicit",
        explicit_stencil([100, 300, 60], fracture_half_length=100), ref,
    )


def test_explicit_three_fracture_stencil_matches_direct_mrst_fixture():
    ref = loadmat(FIXTURE, simplify_cells=True)
    _assert_stencil(
        "explicit3",
        explicit_stencil_3_fracs([100, 300, 60], fracture_half_length=100), ref,
    )
