"""Row and column indices of a block-diagonal layout (MRST ``utils/blockDiagIndex.m``).

Given block heights ``m`` and widths ``n``, returns the `(i, j)` of every entry
of the block-diagonal matrix those blocks form. Building the matrix is then
`sparse(i, j, values)`, which is how MRST assembles per-cell operators without
ever looping over cells.

The two vectors are returned rather than the matrix because the *values* differ
from caller to caller while the sparsity pattern does not - so the pattern is
computed once and reused for the permeability blocks, the projection blocks and
the flux blocks alike.

Called with one argument the blocks are square, which is the common case.
"""

from typing import Optional, Tuple

import numpy as np


def block_diag_index(m, n=None) -> Tuple[np.ndarray, np.ndarray]:
    """Return ``(i, j)`` for blocks of height ``m`` and width ``n``."""
    m = np.atleast_1d(np.asarray(m, dtype=np.int64)).ravel()
    n = m.copy() if n is None else np.atleast_1d(
        np.asarray(n, dtype=np.int64)).ravel()
    if m.size != n.size:
        raise ValueError(
            f"dimension vectors must have the same number of entries, "
            f"got {m.size} and {n.size}")

    # Row offsets of each block, then every row of block b repeated for each
    # of its n[b] columns.
    starts = np.concatenate([[0], np.cumsum(m)])
    rows = np.concatenate([np.tile(np.arange(starts[b], starts[b + 1]), n[b])
                           for b in range(m.size)]) if m.size else \
        np.zeros(0, dtype=np.int64)
    columns = np.repeat(np.arange(int(n.sum())), np.repeat(m, n))
    return rows.astype(np.int64), columns.astype(np.int64)


blockDiagIndex = block_diag_index


if __name__ == "__main__":
    import scipy.sparse as sp

    def _dense(m, n=None, values=None):
        i, j = block_diag_index(m, n)
        values = np.ones(i.size) if values is None else values
        m = np.atleast_1d(m)
        n = m if n is None else np.atleast_1d(n)
        return sp.coo_array((values, (i, j)),
                            shape=(int(m.sum()), int(n.sum()))).toarray()

    def test_one_square_block_is_dense() -> None:
        assert np.all(_dense([2]) == 1), _dense([2])
        print("ok test_one_square_block_is_dense")

    def test_two_blocks_leave_the_corners_empty() -> None:
        out = _dense([2, 2])
        assert out.shape == (4, 4)
        assert np.all(out[:2, :2] == 1) and np.all(out[2:, 2:] == 1)
        assert np.all(out[:2, 2:] == 0) and np.all(out[2:, :2] == 0)
        print("ok test_two_blocks_leave_the_corners_empty")

    def test_rectangular_blocks_are_supported() -> None:
        """The per-cell operators are rarely square: rows are dofs, columns
        are monomials."""
        out = _dense([3, 2], [2, 4])
        assert out.shape == (5, 6), out.shape
        assert np.all(out[:3, :2] == 1) and np.all(out[3:, 2:] == 1)
        assert out.sum() == 3 * 2 + 2 * 4
        print("ok test_rectangular_blocks_are_supported")

    def test_uneven_block_sizes_are_supported() -> None:
        out = _dense([1, 3, 2])
        assert out.shape == (6, 6)
        assert out.sum() == 1 + 9 + 4
        print("ok test_uneven_block_sizes_are_supported")

    def test_a_single_argument_means_square_blocks() -> None:
        first = block_diag_index([2, 3])
        second = block_diag_index([2, 3], [2, 3])
        assert np.array_equal(first[0], second[0])
        assert np.array_equal(first[1], second[1])
        print("ok test_a_single_argument_means_square_blocks")

    def test_the_indices_cover_every_block_entry_exactly_once() -> None:
        i, j = block_diag_index([2, 3], [3, 1])
        assert i.size == 2 * 3 + 3 * 1
        assert len({(int(a), int(b)) for a, b in zip(i, j)}) == i.size
        print("ok test_the_indices_cover_every_block_entry_exactly_once")

    def test_values_land_in_the_order_they_are_given() -> None:
        """Column-major within each block, as MRST's sparse() expects."""
        i, j = block_diag_index([2], [2])
        out = sp.coo_array((np.array([1.0, 2.0, 3.0, 4.0]), (i, j)),
                           shape=(2, 2)).toarray()
        assert np.allclose(out, [[1.0, 3.0], [2.0, 4.0]]), out
        print("ok test_values_land_in_the_order_they_are_given")

    def test_mismatched_vectors_are_rejected() -> None:
        try:
            block_diag_index([1, 2], [1, 2, 3])
        except ValueError as error:
            assert "same number of entries" in str(error)
            print("ok test_mismatched_vectors_are_rejected")
            return
        raise AssertionError("mismatched dimension vectors must be rejected")

    def test_no_blocks_gives_no_indices() -> None:
        i, j = block_diag_index([])
        assert i.size == 0 and j.size == 0
        print("ok test_no_blocks_gives_no_indices")

    test_one_square_block_is_dense()
    test_two_blocks_leave_the_corners_empty()
    test_rectangular_blocks_are_supported()
    test_uneven_block_sizes_are_supported()
    test_a_single_argument_means_square_blocks()
    test_the_indices_cover_every_block_entry_exactly_once()
    test_values_land_in_the_order_they_are_given()
    test_mismatched_vectors_are_rejected()
    test_no_blocks_gives_no_indices()
    print("\nAll tests passed!")
