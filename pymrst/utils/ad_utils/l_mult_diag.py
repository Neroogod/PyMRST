"""Left-multiply Jacobian blocks by a diagonal (MRST ``utils/ad_utils/lMultdiag.m``).

An AD variable's Jacobian is a list of sparse blocks, one per primary variable.
Scaling every row of all of them - by a pore volume, a time step, a mobility -
is `diag(d) @ J` for each block. Written out that way it allocates an `n x n`
sparse matrix and runs a full sparse product per block, which is what MRST
does; scaling the rows directly costs one pass over the stored entries.

The saving is not academic. For a million-cell model with three primary
variables, MRST builds a million-by-million diagonal and performs three sparse
matrix products to achieve what is elementwise multiplication.

Blocks are returned as new matrices; the input list is untouched, since the
caller usually still needs the unscaled Jacobian.
"""

from typing import List, Sequence

import numpy as np
import scipy.sparse as sps


def l_mult_diag(d, blocks: Sequence) -> List:
    """Return ``[diag(d) @ J for J in blocks]``, without forming ``diag(d)``."""
    d = np.asarray(d, dtype=float).ravel()
    out = []
    for index, block in enumerate(blocks):
        if block.shape[0] != d.size:
            raise ValueError(
                f"block {index} has {block.shape[0]} rows but the diagonal has "
                f"{d.size} entries")
        if sps.issparse(block):
            # Scaling the stored entries by their own row's factor is the same
            # product, without building or multiplying by an n-by-n matrix.
            scaled = block.tocsr(copy=True)
            scaled.data *= np.repeat(d, np.diff(scaled.indptr))
            out.append(scaled)
        else:
            out.append(np.asarray(block, dtype=float) * d[:, None])
    return out


lMultdiag = l_mult_diag
lMultDiag = l_mult_diag


if __name__ == "__main__":

    def blocks(rows=6, seed=0):
        rng = np.random.default_rng(seed)
        return [sps.random(rows, 4, density=0.4, random_state=rng, format="csr"),
                sps.random(rows, 7, density=0.3, random_state=rng, format="csr")]

    def test_it_matches_the_explicit_product() -> None:
        """The definition, against the matrix MRST actually builds."""
        d = np.arange(1.0, 7.0)
        given = blocks()
        got = l_mult_diag(d, given)
        expected = [sps.diags(d) @ b for b in given]
        for a, b in zip(got, expected):
            assert np.allclose(a.toarray(), b.toarray())
        print("ok test_it_matches_the_explicit_product")

    def test_each_row_is_scaled_by_its_own_factor() -> None:
        d = np.array([2.0, 0.0, -1.0])
        dense = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        got = l_mult_diag(d, [sps.csr_matrix(dense)])[0].toarray()
        assert np.allclose(got, np.array([[2.0, 4.0], [0.0, 0.0],
                                          [-5.0, -6.0]]))
        print("ok test_each_row_is_scaled_by_its_own_factor")

    def test_every_block_is_scaled() -> None:
        got = l_mult_diag(np.full(6, 3.0), blocks())
        assert len(got) == 2
        for a, b in zip(got, blocks()):
            assert np.allclose(a.toarray(), 3.0 * b.toarray())
        print("ok test_every_block_is_scaled")

    def test_a_dense_block_works_too() -> None:
        dense = np.arange(12.0).reshape(4, 3)
        got = l_mult_diag(np.array([1.0, 2.0, 3.0, 4.0]), [dense])[0]
        assert np.allclose(got, np.diag([1.0, 2.0, 3.0, 4.0]) @ dense)
        print("ok test_a_dense_block_works_too")

    def test_the_input_is_not_modified() -> None:
        """The caller usually still needs the unscaled Jacobian."""
        given = blocks()
        before = [b.toarray().copy() for b in given]
        l_mult_diag(np.full(6, 5.0), given)
        for b, was in zip(given, before):
            assert np.allclose(b.toarray(), was)
        print("ok test_the_input_is_not_modified")

    def test_sparsity_is_preserved() -> None:
        """Scaling rows must not fill in; that is the whole point."""
        given = blocks()
        got = l_mult_diag(np.full(6, 2.0), given)
        for a, b in zip(got, given):
            assert a.nnz == b.nnz, (a.nnz, b.nnz)
        print("ok test_sparsity_is_preserved")

    def test_a_zero_factor_keeps_the_structure() -> None:
        """Zeros stay stored, as they do in the explicit product - a caller
        comparing `nnz` against MRST's result would otherwise see a difference."""
        given = blocks()
        got = l_mult_diag(np.zeros(6), given)
        for a, b in zip(got, given):
            assert a.nnz == b.nnz and np.allclose(a.toarray(), 0.0)
        print("ok test_a_zero_factor_keeps_the_structure")

    def test_an_empty_block_list_gives_an_empty_result() -> None:
        assert l_mult_diag(np.ones(3), []) == []
        print("ok test_an_empty_block_list_gives_an_empty_result")

    def test_a_mismatched_diagonal_is_rejected() -> None:
        try:
            l_mult_diag(np.ones(5), blocks(rows=6))
        except ValueError as error:
            assert "rows but the diagonal" in str(error)
            print("ok test_a_mismatched_diagonal_is_rejected")
            return
        raise AssertionError("a wrong-length diagonal must be rejected")

    def test_it_agrees_with_the_explicit_product_on_a_larger_case() -> None:
        rng = np.random.default_rng(7)
        d = rng.normal(size=400)
        given = [sps.random(400, 250, density=0.01, random_state=rng,
                            format="csr")]
        got = l_mult_diag(d, given)[0]
        expected = sps.diags(d) @ given[0]
        assert abs(got - expected).max() < 1e-12
        print("ok test_it_agrees_with_the_explicit_product_on_a_larger_case")

    test_it_matches_the_explicit_product()
    test_each_row_is_scaled_by_its_own_factor()
    test_every_block_is_scaled()
    test_a_dense_block_works_too()
    test_the_input_is_not_modified()
    test_sparsity_is_preserved()
    test_a_zero_factor_keeps_the_structure()
    test_an_empty_block_list_gives_an_empty_result()
    test_a_mismatched_diagonal_is_rejected()
    test_it_agrees_with_the_explicit_product_on_a_larger_case()
    print("\nAll tests passed!")
