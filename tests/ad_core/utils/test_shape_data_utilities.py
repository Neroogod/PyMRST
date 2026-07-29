import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.backends.diagonal import (
    DiagonalJacobian,
    RowMajorDiagonalJacobian,
)
from pymrst.modules.ad_core.utils import (
    expand_if_uniform,
    expand_matrix_to_cell,
    numel_data,
    size_jac,
)


def test_expansion_helpers_split_columns_and_apply_subset():
    matrix = np.arange(12).reshape(4, 3)
    columns = expand_if_uniform(matrix)
    assert len(columns) == 3
    assert np.array_equal(columns[1], [1, 4, 7, 10])
    subset = expand_matrix_to_cell(matrix, [0, 2])
    assert np.array_equal(subset[0], [0, 6])
    assert np.array_equal(subset[2], [2, 8])
    vector = np.arange(4)
    assert expand_if_uniform(vector) is vector


def test_numel_data_matches_matlab_element_count():
    assert numel_data(np.zeros((2, 3))) == 6
    assert numel_data([1, 2, 3]) == 3
    assert numel_data(4.0) == 1


def test_size_jac_handles_sparse_and_diagonal_backends():
    sparse = sp.csr_matrix((3, 7))
    diagonal = DiagonalJacobian.zeros(4, 9)
    row_major = RowMajorDiagonalJacobian.zeros(5, 11)
    assert size_jac(sparse) == (3, 7)
    assert size_jac(diagonal) == (4, 9)
    assert size_jac(row_major) == (5, 11)
    assert size_jac(diagonal, 1) == 9
