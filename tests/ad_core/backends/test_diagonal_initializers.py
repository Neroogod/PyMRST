"""Tests for MRST diagonal, row-major, and one-block AD initializers."""

import numpy as np

from pymrst.modules.ad_core.backends.diagonal.row_major_diagonal_jacobian import (
    RowMajorDiagonalJacobian,
)
from pymrst.modules.ad_core.backends.diagonal.utils.init_variables_ad_diagonal import (
    initVariablesAD_diagonal,
)
from pymrst.modules.ad_core.backends.diagonal.utils.init_variables_ad_diagonal_row_major import (
    initVariablesAD_diagonalRowMajor,
)
from pymrst.modules.ad_core.backends.diagonal.utils.init_variables_ad_one_block import (
    initVariablesAD_oneBlock,
)


def _dense_blocks(variables):
    return [[block.to_sparse().toarray() for block in var.jac] for var in variables]


def test_default_grouping_matches_consecutive_equal_sizes():
    variables = initVariablesAD_diagonal(
        np.array([1.0, 2.0]),
        np.array([3.0, 4.0]),
        np.array([5.0]),
    )
    blocks = _dense_blocks(variables)
    assert [block.shape for block in variables[0].jac] == [(2, 4), (2, 1)]
    assert np.array_equal(blocks[0][0], [[1, 0, 0, 0], [0, 1, 0, 0]])
    assert np.array_equal(blocks[1][0], [[0, 0, 1, 0], [0, 0, 0, 1]])
    assert np.array_equal(blocks[2][1], [[1]])
    assert not blocks[0][1].any()
    assert not blocks[2][0].any()


def test_explicit_types_group_nonconsecutive_variables():
    variables = initVariablesAD_diagonal(
        np.array([1.0, 2.0]),
        np.array([3.0]),
        np.array([4.0, 5.0]),
        types=[1, 2, 1],
    )
    blocks = _dense_blocks(variables)
    assert np.array_equal(blocks[0][0], [[1, 0, 0, 0], [0, 1, 0, 0]])
    assert np.array_equal(blocks[2][0], [[0, 0, 1, 0], [0, 0, 0, 1]])
    assert np.array_equal(blocks[1][1], [[1]])


def test_row_major_has_identical_sparse_jacobians():
    values = (np.array([1.0, 2.0]), np.array([3.0, 4.0]), np.array([5.0]))
    regular = initVariablesAD_diagonal(*values)
    row_major = initVariablesAD_diagonalRowMajor(*values)
    assert all(
        isinstance(block, RowMajorDiagonalJacobian)
        for variable in row_major
        for block in variable.jac
    )
    for normal, row in zip(regular, row_major):
        for a, b in zip(normal.jac, row.jac):
            assert np.array_equal(a.toarray(), b.toarray())


def test_one_block_uses_global_column_offsets():
    variables = initVariablesAD_oneBlock(
        np.array([1.0, 2.0]), np.array([3.0]), np.array([4.0, 5.0])
    )
    expected = [
        [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
        [[0, 0, 1, 0, 0]],
        [[0, 0, 0, 1, 0], [0, 0, 0, 0, 1]],
    ]
    for variable, matrix in zip(variables, expected):
        assert len(variable.jac) == 1
        assert np.array_equal(variable.jac[0].toarray(), matrix)


def test_group_rejects_mixed_sizes():
    try:
        initVariablesAD_diagonal(
            np.array([1.0, 2.0]), np.array([3.0]), types=[1, 1]
        )
    except ValueError as err:
        assert "equal size" in str(err)
    else:
        raise AssertionError("mixed-size diagonal group was accepted")
