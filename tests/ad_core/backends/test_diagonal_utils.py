"""Tests for MRST diagonal helpers.

Covers ``diagMult``, ``diagProductMult``, ``double2GenericAD``,
``getSparseArguments``, ``getSparseBlocks``, ``incrementSubset``, and
``matrixDims``.
"""

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.backends.diagonal.generic_ad import (
    GenericAD,
    init_variables_diagonal,
)
from pymrst.modules.ad_core.backends.diagonal.utils import (
    diagMult,
    diagProductMult,
    double2GenericAD,
    getSparseArguments,
    getSparseBlocks,
    incrementSubset,
    matrixDims,
)


def test_diag_product_mult_matches_definition():
    x = sp.csr_array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    y = sp.csr_array([[6.0, 5.0], [4.0, 3.0], [2.0, 1.0]])
    v1 = np.array([2.0, 3.0, 4.0])
    v2 = np.array([5.0, 6.0, 7.0])
    result, _, _ = diagProductMult(v1, v2, x, y)
    expected = np.diag(v2) @ x.toarray() + np.diag(v1) @ y.toarray()
    assert np.array_equal(result.toarray(), expected)


def test_double_to_generic_ad_has_zero_derivatives():
    (sample,) = init_variables_diagonal(np.array([1.0, 2.0, 3.0]))
    converted = double2GenericAD(np.array([7.0, 8.0]), sample)
    assert isinstance(converted, GenericAD)
    assert np.array_equal(converted.val, [7.0, 8.0])
    assert converted.jac[0].shape == (2, 3)
    assert converted.jac[0].to_sparse().nnz == 0
    assert np.array_equal(double2GenericAD([4.0], np.array([1.0])), [4.0])


def test_get_sparse_blocks_forwards_triplets():
    matrix = sp.csr_array([[1.0, 0.0], [2.0, 3.0]])
    row, col, data = getSparseBlocks(matrix, ioffset=2, joffset=4)
    rebuilt = sp.csr_array((data, (row, col)), shape=(4, 6)).toarray()
    assert np.array_equal(rebuilt[2:4, 4:6], matrix.toarray())


def test_increment_subset_numeric_and_ad():
    numeric = incrementSubset(np.array([1.0, 2.0, 3.0]), [0, 2], 5.0)
    assert np.array_equal(numeric, [6.0, 2.0, 8.0])

    (ad,) = init_variables_diagonal(np.array([1.0, 2.0, 3.0]))
    updated = incrementSubset(ad, [0, 2], 5.0)
    assert np.array_equal(updated.val, [6.0, 2.0, 8.0])
    assert np.array_equal(updated.to_adi().jac[0].toarray(), np.eye(3))
    assert np.array_equal(ad.val, [1.0, 2.0, 3.0])


def test_diag_mult_fast_paths_and_row_scaling():
    matrix = sp.csr_array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    scale = np.array([2.0, 0.0, -1.0])
    assert np.array_equal(
        diagMult(scale, matrix).toarray(),
        np.diag(scale) @ matrix.toarray(),
    )
    assert diagMult(np.zeros(3), matrix).nnz == 0
    assert np.array_equal(
        diagMult([3.0], matrix).toarray(), 3.0 * matrix.toarray()
    )


def test_sparse_arguments_offsets_rebuild_original_block():
    matrix = sp.csr_array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    row, col, data = getSparseArguments(matrix, ioffset=3, joffset=2)
    rebuilt = sp.csr_array((data, (row, col)), shape=(6, 7)).toarray()
    assert np.array_equal(rebuilt[3:5, 2:5], matrix.toarray())
    assert rebuilt.sum() == matrix.sum()


def test_matrix_dims_accepts_diagonal_sparse_and_dense_storage():
    (variable,) = init_variables_diagonal(np.arange(4.0))
    assert matrixDims(variable.jac[0]) == (4, 4)
    assert matrixDims(sp.csr_array((2, 5))) == (2, 5)
    assert matrixDims(np.zeros((3, 7))) == (3, 7)
