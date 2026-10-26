"""MRST ``ad-core/backends/diagonal/utils/matrixDims.m``: MATLAB's `size` of a
matrix or Jacobian block - both dimensions, or the one asked for (1-based, as
MATLAB's `size(M, dim)`)."""

from __future__ import annotations

from typing import Any, Optional, Tuple, Union

import numpy as np


def matrix_dims(M: Any, dim: Optional[int] = None) -> Union[int, Tuple[int, int]]:
    """MRST `[m, n] = matrixDims(M)` or `matrixDims(M, dim)`."""
    shape = tuple(M.shape) if hasattr(M, "shape") else np.shape(M)
    shape = shape + (1,)*(2 - len(shape))
    if dim is None:
        return shape
    return shape[dim - 1] if dim <= len(shape) else 1


matrixDims = matrix_dims
__all__ = ["matrixDims", "matrix_dims"]


if __name__ == "__main__":
    import scipy.sparse as sp

    def test_size_and_one_dimension():
        assert matrix_dims(sp.csr_array((2, 5))) == (2, 5) and matrix_dims(np.zeros((3, 7)), 2) == 7
        assert matrix_dims(np.zeros(4)) == (4, 1) and matrix_dims(np.zeros((2, 2)), 3) == 1
        print("ok test_size_and_one_dimension")

    test_size_and_one_dimension()
    print("\nAll tests passed!")
