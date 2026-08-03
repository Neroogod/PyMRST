"""Numerical parity with MRST 2024a for deterministic root utilities."""

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from pymrst.utils import (
    block_diag_index,
    dinterp_table,
    dinterpq1,
    discspace1,
    format_time_range,
    interp_table,
    numel_value,
    reduce_to_double,
    rlencode,
    subset_minus,
    subset_plus,
    unique_stable,
)


REF = loadmat(Path(__file__).parents[2] / "scripts/matlab/ref_utils_core.mat",
              squeeze_me=True)


def vector(name):
    return np.asarray(REF[name]).ravel()


def test_block_diag_index_matches_mrst():
    i, j = block_diag_index(vector("m"), vector("n"))
    np.testing.assert_array_equal(i, vector("block_i") - 1)
    np.testing.assert_array_equal(j, vector("block_j") - 1)


def test_table_interpolation_and_derivatives_match_mrst():
    x, y, xi = vector("x"), vector("y"), vector("xi")
    np.testing.assert_allclose(dinterpq1(x, y, xi), vector("dq"))
    out, derivative = interp_table(x, y, xi, return_derivative=True)
    np.testing.assert_allclose(out, vector("linear_y"))
    np.testing.assert_allclose(derivative, vector("linear_d"))
    np.testing.assert_allclose(dinterp_table(x, y, xi), vector("linear_d"))
    out, derivative = interp_table(x, y, xi, spline=True,
                                   return_derivative=True)
    np.testing.assert_allclose(out, vector("spline_y"), rtol=1e-12)
    np.testing.assert_allclose(derivative, vector("spline_d"), rtol=1e-12)
    np.testing.assert_allclose(dinterp_table(x, y, xi, spline=True),
                               vector("spline_d"), rtol=1e-12)
    np.testing.assert_allclose(interp_table(x[::-1], y[::-1], xi),
                               vector("linear_y"))


def test_discspace1_matches_mrst():
    refine = np.asarray(REF["refine"])
    np.testing.assert_allclose(discspace1(0, 10, 9), vector("disc_plain"))
    np.testing.assert_allclose(discspace1(0, 10, 17, refine),
                               vector("disc_linear"))
    np.testing.assert_allclose(discspace1(0, 10, 17, refine, "constant"),
                               vector("disc_constant"))


def test_format_time_range_matches_mrst():
    assert format_time_range(2*86400 + 3*3600 + 4.005) == REF["time_unlimited"]
    assert format_time_range(2*86400 + 3*3600 + 4, 2) == REF["time_limited"]


def test_rlencode_matches_mrst_along_rows_and_columns():
    values, counts = rlencode(REF["run_rows_in"], dim=0)
    np.testing.assert_array_equal(values, REF["run_rows"])
    np.testing.assert_array_equal(counts, vector("run_rows_n"))
    values, counts = rlencode(REF["run_cols_in"], dim=1)
    np.testing.assert_array_equal(values, REF["run_cols"])
    np.testing.assert_array_equal(counts, vector("run_cols_n"))


def test_subset_numel_and_reduce_helpers_match_mrst():
    np.testing.assert_array_equal(
        subset_plus(np.zeros((4, 2)), [[1, 2], [3, 4]], [1, 3]),
        REF["subset_plus_out"])
    source = np.arange(1, 7).reshape((3, 2), order="F")
    np.testing.assert_array_equal(subset_minus(source, [10, 20], [1, 4]),
                                  REF["subset_minus_out"])
    assert numel_value(np.zeros((2, 3))) == int(REF["numel_out"])
    value = np.arange(1, 7).reshape((2, 3), order="F")
    assert reduce_to_double(value) is value
    np.testing.assert_array_equal(value, REF["reduce_out"])


def test_unique_stable_elements_and_rows_match_mrst():
    source = REF["unique_in"]
    c, ia, ic = unique_stable(source)
    np.testing.assert_array_equal(c, vector("unique_values"))
    np.testing.assert_array_equal(ia, vector("unique_ia") - 1)
    np.testing.assert_array_equal(ic, vector("unique_ic") - 1)
    c, ia, ic = unique_stable(source, rows=True)
    np.testing.assert_array_equal(c, REF["unique_rows"])
    np.testing.assert_array_equal(ia, vector("unique_rows_ia") - 1)
    np.testing.assert_array_equal(ic, vector("unique_rows_ic") - 1)


def test_unique_stable_rows_treats_a_vector_as_one_row():
    source = np.array([3, 1, 3])
    c, ia, ic = unique_stable(source, rows=True)
    np.testing.assert_array_equal(c, source)
    np.testing.assert_array_equal(ia, [0])
    np.testing.assert_array_equal(ic, [0])
