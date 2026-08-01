"""Collapse a block-diagonal matrix (MRST ``vem/utils/squeezeBlockDiag.m``).

VEM stores per-cell operators stacked block-diagonally: one block per cell,
with block `i` having `n[i]` rows or columns. Most of the time that is what you
want. Sometimes you want the blocks laid side by side instead - all cells'
contributions to the same local index in one column - and that is what this
does, by multiplying with a tiled identity.

Which direction it collapses is decided by ``r``: if ``r`` equals the matrix's
own row count the columns are folded, otherwise the rows are. MRST infers the
direction this way rather than taking a flag, so passing the wrong ``r``
silently collapses the other axis instead of raising. This port keeps the
inference - callers in `incompVEM` rely on it - but rejects an ``r`` that
matches neither dimension, which MRST does not.

The operation is exactly `A @ tile(I(c), N, 1)` or `tile(I(r), 1, N) @ A`; it
stays sparse throughout, because these matrices have one row per degree of
freedom in the grid.
"""

import numpy as np
import scipy.sparse as sp


def squeeze_block_diag(A, n, r: int, c: int):
    """Fold ``A``'s blocks together, along whichever axis ``r`` selects.

    ``n`` gives the block sizes (only its length is used, as in MRST); ``r``
    and ``c`` are the collapsed shape.
    """
    A = sp.csr_array(A) if not sp.issparse(A) else A
    rows, columns = A.shape
    blocks = len(np.atleast_1d(n))

    if r == rows:
        return A @ sp.vstack([sp.eye(c, format="csr")] * blocks, format="csr")
    if c == columns:
        return sp.hstack([sp.eye(r, format="csr")] * blocks, format="csr") @ A
    raise ValueError(
        f"neither r={r} nor c={c} matches A's shape {A.shape}; "
        f"squeezeBlockDiag infers its direction from them")


squeezeBlockDiag = squeeze_block_diag


if __name__ == "__main__":

    def _blocks(mats):
        return sp.block_diag(mats, format="csr")

    def test_columns_fold_when_r_matches_the_row_count() -> None:
        """Three 2x2 blocks stacked down the diagonal, folded to 6x2."""
        mats = [np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([[5.0, 6.0], [7.0, 8.0]]),
                np.array([[9.0, 10.0], [11.0, 12.0]])]
        A = _blocks(mats)
        out = squeeze_block_diag(A, [2, 2, 2], A.shape[0], 2)
        assert out.shape == (6, 2), out.shape
        assert np.allclose(out.toarray(), np.vstack(mats))
        print("ok test_columns_fold_when_r_matches_the_row_count")

    def test_rows_fold_when_c_matches_the_column_count() -> None:
        mats = [np.array([[1.0, 2.0], [3.0, 4.0]]),
                np.array([[5.0, 6.0], [7.0, 8.0]])]
        A = _blocks(mats)
        out = squeeze_block_diag(A, [2, 2], 2, A.shape[1])
        assert out.shape == (2, 4), out.shape
        assert np.allclose(out.toarray(), np.hstack(mats))
        print("ok test_rows_fold_when_c_matches_the_column_count")

    def test_the_direction_comes_from_r_not_from_a_flag() -> None:
        """A square input is folded one way or the other purely by r and c."""
        A = _blocks([np.eye(2), 2 * np.eye(2)])
        down = squeeze_block_diag(A, [2, 2], 4, 2)
        across = squeeze_block_diag(A, [2, 2], 2, 4)
        assert down.shape == (4, 2) and across.shape == (2, 4)
        print("ok test_the_direction_comes_from_r_not_from_a_flag")

    def test_only_the_block_count_of_n_is_used() -> None:
        """MRST takes numel(n); the individual sizes never enter."""
        A = _blocks([np.eye(2), 2 * np.eye(2)])
        first = squeeze_block_diag(A, [2, 2], 4, 2).toarray()
        second = squeeze_block_diag(A, [99, 7], 4, 2).toarray()
        assert np.allclose(first, second)
        print("ok test_only_the_block_count_of_n_is_used")

    def test_the_result_stays_sparse() -> None:
        A = _blocks([np.eye(3)] * 4)
        out = squeeze_block_diag(A, [3] * 4, A.shape[0], 3)
        assert sp.issparse(out), type(out)
        print("ok test_the_result_stays_sparse")

    def test_a_single_block_is_returned_unchanged() -> None:
        M = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = squeeze_block_diag(sp.csr_array(M), [2], 2, 2)
        assert np.allclose(out.toarray(), M)
        print("ok test_a_single_block_is_returned_unchanged")

    def test_uneven_blocks_fold_correctly() -> None:
        """Blocks need not be the same size, only the folded width shared."""
        mats = [np.arange(4.0).reshape(2, 2), np.arange(6.0).reshape(3, 2)]
        A = _blocks(mats)
        out = squeeze_block_diag(A, [2, 3], A.shape[0], 2)
        assert np.allclose(out.toarray(), np.vstack(mats))
        print("ok test_uneven_blocks_fold_correctly")

    def test_a_shape_matching_neither_dimension_is_rejected() -> None:
        """MRST would silently fold the other axis; this refuses."""
        A = _blocks([np.eye(2), np.eye(2)])
        try:
            squeeze_block_diag(A, [2, 2], 7, 9)
        except ValueError as error:
            assert "infers its direction" in str(error)
            print("ok test_a_shape_matching_neither_dimension_is_rejected")
            return
        raise AssertionError("an impossible shape must be rejected")

    test_columns_fold_when_r_matches_the_row_count()
    test_rows_fold_when_c_matches_the_column_count()
    test_the_direction_comes_from_r_not_from_a_flag()
    test_only_the_block_count_of_n_is_used()
    test_the_result_stays_sparse()
    test_a_single_block_is_returned_unchanged()
    test_uneven_blocks_fold_correctly()
    test_a_shape_matching_neither_dimension_is_rejected()
    print("\nAll tests passed!")
