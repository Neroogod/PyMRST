"""MRST `DiagonalJacobian` accessors, ported by meaning (Phase 2).

MRST stores an `n x ndiag` array plus an optional `subset` and a `rowMajor`
flag. This class stores nonzero `(data, row, col)` triplets - a deliberate
choice its own docstring explains, because TPFA operators produce shifted
diagonals a strict diagonal array cannot hold.

So these methods are ported by what they *ask*, not by transcription. Where
MRST's question survives the change of representation - is this block zero,
which rows does it touch, do two blocks share a pattern - it is implemented and
tested here. `compareIndices` and `treatSubset` only mean something against the
array layout, and transcribing them would invent semantics, so they are absent
and recorded in the ledger.

The distinction the first test draws is the one that matters: *structurally*
zero and *numerically* zero are different states, and they assemble
differently.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from pymrst.modules.ad_core.backends.diagonal.diagonal_jacobian import (
    DiagonalJacobian,
)


def _two_variable_block():
    """One block spanning two primary variables: `[diag(1,2) | diag(3,4)]`."""
    return DiagonalJacobian.from_sparse(
        np.hstack([np.diag([1.0, 2.0]), np.diag([3.0, 4.0])]))


# -- zero, in two different senses -----------------------------------------

def test_structurally_zero_is_not_the_same_as_numerically_zero():
    """A block with no entries contributes no sparsity pattern; a block of
    explicit zeros does. Collapsing the two changes what assembly produces."""
    empty = DiagonalJacobian.zeros(3, 3)
    explicit = empty.expand_zero()

    assert empty.is_zero()
    assert not explicit.is_zero(), "explicit zeros are a pattern, not absence"
    assert explicit.nnz == 9
    assert not np.any(explicit.data)


def test_is_all_zeros_looks_at_values_not_structure():
    assert DiagonalJacobian.is_all_zeros(np.zeros(4))
    assert not DiagonalJacobian.is_all_zeros(np.array([0.0, 1.0]))
    assert DiagonalJacobian.is_all_zeros(DiagonalJacobian.zeros(2, 2))
    assert not DiagonalJacobian.is_all_zeros(DiagonalJacobian.identity(2))


def test_to_zero_can_resize():
    block = DiagonalJacobian.identity(3)
    assert block.to_zero().shape == (3, 3)
    assert block.to_zero(5).shape == (5, 3)
    assert block.to_zero(5).is_zero()


# -- which rows a block touches --------------------------------------------

def test_the_subset_is_the_rows_that_carry_entries():
    block = DiagonalJacobian(np.array([1.0, 2.0]), np.array([0, 2]),
                              np.array([0, 2]), (4, 4))
    np.testing.assert_array_equal(block.get_subset(), [0, 2])
    assert DiagonalJacobian.zeros(4, 4).get_subset().size == 0


def test_the_subset_is_sorted_and_deduplicated():
    """Two entries in one row is one row, and order must not depend on how the
    triplets happened to be built."""
    block = DiagonalJacobian(np.ones(3), np.array([2, 0, 2]),
                              np.array([0, 1, 2]), (3, 3))
    np.testing.assert_array_equal(block.get_subset(), [0, 2])


def test_an_empty_block_shares_a_pattern_with_anything():
    """MRST's rule, and it is a performance decision as much as a semantic
    one: without it every operation involving a zero block takes the slow
    path."""
    empty = DiagonalJacobian.zeros(3, 3)
    assert empty.subsets_equal(DiagonalJacobian.identity(3))
    assert DiagonalJacobian.identity(3).subsets_equal(empty)


def test_two_blocks_touching_different_rows_do_not_share_a_pattern():
    first = DiagonalJacobian(np.ones(1), np.array([0]), np.array([0]), (3, 3))
    second = DiagonalJacobian(np.ones(1), np.array([1]), np.array([1]), (3, 3))
    assert not first.subsets_equal(second)
    assert first.subsets_equal(first)


def test_the_zero_check_can_be_skipped_explicitly():
    """`subsetsEqualNoZeroCheck` compares patterns literally, so an empty block
    no longer matches everything."""
    empty = DiagonalJacobian.zeros(3, 3)
    assert not empty.subsets_equal_no_zero_check(DiagonalJacobian.identity(3))


# -- reading derivatives out ------------------------------------------------

def test_a_block_spanning_two_variables_is_indexed_per_variable():
    """Adjacent column groups of `nrow` each. Reading the wrong group returns
    another variable's derivatives, which is arithmetically valid and wrong."""
    block = _two_variable_block()
    np.testing.assert_allclose(block.get_diagonal_by_index(0), [1.0, 2.0])
    np.testing.assert_allclose(block.get_diagonal_by_index(1), [3.0, 4.0])


def test_indexing_past_the_last_variable_is_refused():
    with pytest.raises(IndexError, match="of 2"):
        _two_variable_block().get_diagonal_by_index(2)


def test_a_width_that_is_not_a_whole_number_of_blocks_is_refused():
    """Guessing the split would silently mix two variables' derivatives."""
    ragged = DiagonalJacobian.zeros(2, 3)
    with pytest.raises(ValueError, match="whole number"):
        ragged.get_diagonal_by_index(0)


def test_the_main_diagonal_can_be_read_whole_or_in_part():
    block = DiagonalJacobian.from_sparse(np.diag([5.0, 6.0, 7.0]))
    np.testing.assert_allclose(block.get_diagonal(), [5.0, 6.0, 7.0])
    np.testing.assert_allclose(block.get_diagonal([0, 2]), [5.0, 7.0])


def test_the_diagonal_count_is_the_derivative_width():
    assert DiagonalJacobian.zeros(4, 12).get_number_of_diagonals() == 12


# -- tiling -----------------------------------------------------------------

def test_tiling_returns_sparse_because_a_tiled_diagonal_is_not_diagonal():
    tiled = DiagonalJacobian.identity(2).repmat(2, 3)
    assert tiled.shape == (4, 6)
    np.testing.assert_allclose(tiled.toarray(),
                               np.tile(np.eye(2), (2, 3)))
