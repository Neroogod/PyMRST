"""MATLAB references for the remaining deterministic utils numerics."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.io import loadmat

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.utils import (
    cell_dims_cg,
    get_sorted_cell_nodes,
    incomp_hydrostatic_pressure,
    is_coarse_grid,
)
from pymrst.utils.ad_utils.l_mult_diag import l_mult_diag
from pymrst.utils.units.get_unit_system import get_unit_system


REF = loadmat(Path(__file__).parents[2] /
              "scripts/matlab/ref_utils_remaining_numeric.mat", squeeze_me=True)


def vec(name, dtype=None):
    return np.asarray(REF[name], dtype=dtype).ravel()


def test_l_multdiag_matches_mrst_sparse_products():
    blocks = [REF["jac_a"].tocsr(), REF["jac_b"].tocsr()]
    got = l_mult_diag(vec("diag_values"), blocks)
    np.testing.assert_allclose(got[0].toarray(), REF["scaled_a"])
    np.testing.assert_allclose(got[1].toarray(), REF["scaled_b"])


def test_cell_dims_cg_matches_mrst():
    grid = compute_geometry(cart_grid([3, 2], [6.0, 10.0]))
    got = cell_dims_cg(grid, vec("dims_cells", int) - 1)
    for values, name in zip(got, ("dims_x", "dims_y", "dims_z")):
        np.testing.assert_allclose(values, vec(name))


def test_get_sorted_cell_nodes_matches_mrst_orientation():
    grid = compute_geometry(cart_grid([3, 2], [6.0, 10.0]))
    np.testing.assert_array_equal(get_sorted_cell_nodes(grid) + 1,
                                  vec("sorted_cell_nodes"))


def test_incomp_hydrostatic_pressure_matches_mrst_sampling():
    grid = compute_geometry(cart_grid([1, 1, 8], [10.0, 10.0, 80.0]))
    pressure, fn = incomp_hydrostatic_pressure(
        grid, vec("contacts"), vec("densities"),
        top_pressure=1.25e5, n_samples=41)
    np.testing.assert_allclose(pressure, vec("hydro_p"), rtol=1e-12)
    np.testing.assert_allclose(fn(vec("hydro_query_z")),
                               vec("hydro_query_p"), rtol=1e-12)


def test_is_coarse_grid_matches_mrst_field_presence_contract():
    assert is_coarse_grid(SimpleNamespace()) == bool(REF["coarse_plain"])
    assert is_coarse_grid(SimpleNamespace(parent=None, partition=None)) == \
        bool(REF["coarse_both"])
    assert is_coarse_grid(SimpleNamespace(parent=None)) == \
        bool(REF["coarse_parent"])


def test_get_unit_system_matches_all_mrst_fields():
    fields = ("length", "time", "press", "viscosity", "perm",
              "resvolume", "trans")
    for row, name in zip(np.asarray(REF["unit_values"]),
                         ("metric", "field", "lab", "si")):
        system = get_unit_system(name)
        np.testing.assert_allclose([getattr(system, field) for field in fields],
                                   row, rtol=1e-14)
