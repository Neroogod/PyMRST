"""MRST ``ad-core/backends/diagonal/utils/getSparseArguments.m``: the nonzero
triplets of a matrix in MATLAB's `find` order (column by column), shifted by
the offsets, and its size."""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np
import scipy.sparse as sp


def get_sparse_arguments(M: Any, ioffset: int = 0, joffset: int = 0,
                         nargout: int = 3) -> Tuple[Any, ...]:
    """MRST `[I, J, V, m, n] = getSparseArguments(M, ioffset, joffset)`, 0-based."""
    if hasattr(M, "to_sparse"):
        M = M.to_sparse()
    C = sp.csc_array(M)
    C.eliminate_zeros()
    C.sort_indices()
    J = np.repeat(np.arange(C.shape[1]), np.diff(C.indptr))
    I = C.indices.astype(np.int64) + ioffset
    V = C.data.copy()
    out = (I, J + joffset, V, C.shape[0], C.shape[1])
    return out[:nargout]


getSparseArguments = get_sparse_arguments
__all__ = ["getSparseArguments", "get_sparse_arguments"]


if __name__ == "__main__":
    def test_column_major_order_offsets_and_size():
        M = np.array([[0.0, 2.0], [3.0, 0.0], [4.0, 5.0]])
        I, J, V, m, n = get_sparse_arguments(M, 10, 20, nargout=5)
        assert I.tolist() == [11, 12, 10, 12] and J.tolist() == [20, 20, 21, 21]
        assert V.tolist() == [3.0, 4.0, 2.0, 5.0] and (m, n) == (3, 2)
        print("ok test_column_major_order_offsets_and_size")

    def test_explicit_zeros_are_not_returned():
        M = sp.csr_array((np.array([0.0, 1.0]), (np.array([0, 1]), np.array([0, 1]))), shape=(2, 2))
        I, J, V = get_sparse_arguments(M)
        assert V.tolist() == [1.0] and I.tolist() == [1]
        print("ok test_explicit_zeros_are_not_returned")

    test_column_major_order_offsets_and_size()
    test_explicit_zeros_are_not_returned()
    print("\nAll tests passed!")
