"""Direct MRST certification of the shale EDFM/pEDFM operator pipeline."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.shale.data import FracturePlane
from pymrst.modules.shale.hfm_fns import (
    edfm_shale_grid, setup_shale_edfm_operators_tpfa,
)
from pymrst.modules.shale.other_mrst_fns import make_shale_rock
from pymrst.modules.shale.pedfm_nnc import (
    projected_matrix_fracture_nncs_3d, setup_pedfm_operators_tpfa,
)
from pymrst.solvers import get_face_transmissibility
from pymrst.utils.units import DARCY, MILLI


FIXTURE = Path(__file__).parents[2]/"tests/fixtures/shale/operator_pipeline.mat"


def _pipeline():
    grid = compute_geometry(cart_grid([3,3,3], [3.,3.,3.]))
    rock = make_shale_rock(grid, 100*MILLI*DARCY, .2)
    plane = FracturePlane(
        np.array([[1.5,.75,.75], [1.5,2.25,.75],
                  [1.5,2.25,2.25], [1.5,.75,2.25]]),
        aperture=.01, porosity=.5, permeability=10*DARCY,
    )
    global_grid, _ = edfm_shale_grid(grid, [plane], rock, tolerance=1e-5)
    face_t = get_face_transmissibility(global_grid.grid, global_grid.rock)
    volumes = global_grid.grid.cells.volumes
    edfm = setup_shale_edfm_operators_tpfa(
        global_grid.grid, global_grid.rock, face_t, global_grid.nnc,
        cell_volumes=volumes,
    )
    projected = projected_matrix_fracture_nncs_3d(
        global_grid.edfm, global_grid.fractures, global_grid.projections,
        tolerance=1e-5,
    )
    pedfm = setup_pedfm_operators_tpfa(
        global_grid.grid, global_grid.rock, face_t, global_grid.nnc,
        projected=projected, tolerance=1e-5, cell_volumes=volumes,
    )
    return global_grid, projected, edfm, pedfm


def _connection_map(neighbors, transmissibility, *, matlab=False):
    shift = 1 if matlab else 0
    return {
        tuple(sorted((int(pair[0])-shift, int(pair[1])-shift))): float(value)
        for pair, value in zip(neighbors, transmissibility)
    }


def test_global_edfm_topology_and_nnc_match_direct_mrst_pipeline():
    ref = loadmat(FIXTURE, simplify_cells=True)
    global_grid, _, _, _ = _pipeline()
    assert global_grid.n_matrix == int(ref["matrix_cells"])
    assert global_grid.n_cells == int(ref["edfm_cells"])
    expected = np.asarray(ref["edfm_nnc_cells"], dtype=int)-1
    assert {tuple(row) for row in global_grid.nnc.cells} == {tuple(row) for row in expected}
    actual_t = _connection_map(global_grid.nnc.cells, global_grid.nnc.T)
    expected_t = _connection_map(expected, ref["edfm_nnc_T"])
    assert actual_t.keys() == expected_t.keys()
    for pair in actual_t:
        assert np.isclose(actual_t[pair], expected_t[pair], rtol=2e-12)


def test_edfm_and_pedfm_active_operators_match_direct_mrst_pipeline():
    ref = loadmat(FIXTURE, simplify_cells=True)
    _, _, edfm, pedfm = _pipeline()
    for prefix, operators in (("edfm", edfm), ("pedfm", pedfm)):
        expected = _connection_map(
            np.asarray(ref[f"{prefix}_N"]), ref[f"{prefix}_T"], matlab=True,
        )
        actual = _connection_map(operators.N, operators.T)
        assert actual.keys() == expected.keys()
        if prefix == "edfm":
            for pair in actual:
                assert np.isclose(
                    actual[pair], expected[pair], rtol=2e-12, atol=1e-30,
                )
        else:
            # The full topology is certified here. MRST 2024a stores projected
            # areas and projected transmissibilities on inconsistent rows for
            # this case; the physical-invariant test below tracks this known
            # reference defect explicitly instead of copying it silently.
            assert len(actual) == 96
        np.testing.assert_allclose(
            operators.pv, np.asarray(ref[f"{prefix}_pv"]), rtol=2e-12,
        )


def test_pedfm_projection_matches_mrst_topology_and_repairs_legacy_t_rows():
    """Certify the projection while documenting a MRST 2024a row defect.

    MRST selects exactly the same MF/MM rows and areas.  Its saved ``G.nnc.T``
    nevertheless contains zero transmissibility on positive-area projected
    connections and nonzero transmissibility on zero-area connections.  The
    PyMRST owner preserves the required invariant T > 0 iff projected area > 0.
    """
    ref = loadmat(FIXTURE, simplify_cells=True)
    _, projected, _, _ = _pipeline()
    expected_mf = np.atleast_2d(ref["pedfm_projected_MF"]).astype(int) - 1
    expected_mm = np.atleast_2d(ref["pedfm_projected_MM"]).astype(int) - 1
    expected_area = np.atleast_1d(ref["pedfm_projected_area"])
    legacy_t = np.atleast_1d(ref["pedfm_projected_T"])

    np.testing.assert_array_equal(projected.MFcells, expected_mf)
    np.testing.assert_array_equal(projected.MMneighs, expected_mm)
    np.testing.assert_allclose(projected.area, expected_area, rtol=2e-12)
    assert np.all(projected.T[projected.area > 0] > 0)
    assert np.all(projected.T[projected.area == 0] == 0)

    # This is the precise reason corrected PyMRST transmissibilities cannot be
    # equal to the malformed legacy vector row-for-row.
    assert np.any((expected_area > 0) & (legacy_t == 0))
    assert np.any((expected_area == 0) & (legacy_t > 0))


def test_pedfm_matrix_face_multipliers_match_direct_mrst_pipeline():
    """Compare every active grid-face T before projected NNC rows begin."""
    ref = loadmat(FIXTURE, simplify_cells=True)
    _, projected, _, pedfm = _pipeline()
    n_projected = projected.MFcells.shape[0]
    n_base_nnc = 9
    n_grid_internal = pedfm.N.shape[0] - n_base_nnc - n_projected
    actual = _connection_map(
        pedfm.N[:n_grid_internal], pedfm.T[:n_grid_internal],
    )
    expected = _connection_map(
        np.asarray(ref["pedfm_N"])[:n_grid_internal],
        np.asarray(ref["pedfm_T"])[:n_grid_internal],
        matlab=True,
    )
    assert actual.keys() == expected.keys()
    for pair in actual:
        assert np.isclose(actual[pair], expected[pair], rtol=2e-12, atol=1e-30)
