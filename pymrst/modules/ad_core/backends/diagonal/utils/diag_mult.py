"""MRST ``ad-core/backends/diagonal/utils/diagMult.m``: `diag(v)*M`, with an
empty result for a zero `v`, `M` itself when it has no nonzeros, `v*M` for a
scalar `v`, and the diagonal matrix `D` built once and returned for reuse."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np
import scipy.sparse as sp


def diag_mult(v: Any, M: Any, D: Optional[Any] = None,
              nargout: int = 1) -> Union[Any, Tuple[Any, Any]]:
    """MRST `[x, D] = diagMult(v, M, D)`; `nargout=2` also returns `D`."""
    v = np.asarray(v, dtype=float).ravel()
    M = sp.csr_array(M)
    if not np.any(v):
        x = sp.csr_array(M.shape)
    elif M.nnz == 0:
        x = M
    elif v.size == 1:
        x = (v[0]*M).tocsr()
    else:
        if D is None:
            n = v.size
            D = sp.csr_array((v, (np.arange(n), np.arange(n))), shape=(n, n))
        x = (sp.csr_array(D) @ M).tocsr()
    return (x, D) if nargout > 1 else x


diagMult = diag_mult
__all__ = ["diagMult", "diag_mult"]


if __name__ == "__main__":
    M = sp.csr_array(np.arange(1.0, 7.0).reshape(3, 2))

    def test_scaling_and_the_reusable_diagonal():
        x, D = diag_mult(np.array([1.0, 2.0, 3.0]), M, nargout=2)
        assert np.allclose(x.toarray(), np.diag([1, 2, 3]) @ M.toarray()) and D.shape == (3, 3)
        assert np.allclose(diag_mult(np.array([9.0, 9.0, 9.0]), M, D).toarray(), x.toarray())
        print("ok test_scaling_and_the_reusable_diagonal")

    def test_fast_paths():
        assert diag_mult(np.zeros(3), M).nnz == 0
        assert np.allclose(diag_mult(np.array([2.0]), M).toarray(), 2*M.toarray())
        empty = sp.csr_array((3, 2))
        assert diag_mult(np.ones(3), empty) is not None and diag_mult(np.ones(3), empty).nnz == 0
        print("ok test_fast_paths")

    test_scaling_and_the_reusable_diagonal()
    test_fast_paths()
    print("\nAll tests passed!")
