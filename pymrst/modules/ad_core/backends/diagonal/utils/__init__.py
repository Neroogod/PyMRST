"""Diagonal-backend numeric helpers (MRST `backends/diagonal/utils`).

Small, layout-independent primitives the diagonal backend's assembly reaches
for: pull the `(I, J, V)` triplet out of a Jacobian block for block assembly
(`get_sparse_arguments`), left-multiply a matrix by a diagonal without forming
the diagonal when it is trivial (`diag_mult`), and report a matrix's dimensions
(`matrix_dims`). These mirror MRST's `getSparseArguments`, `diagMult` and
`matrixDims`.

The block-layout members of `backends/diagonal` - `FixedWidthJacobian`,
`ConservationLawJacobian`, `blockCSRtoSparse` - are deliberately not ported here:
their speed-up lives in the MEX block kernels (`operators/mex/`, explicitly out
of scope), and `blockCSRtoSparse` untransposes that MEX-specific block storage,
so it has no meaning without the MEX path. In pure NumPy those layouts add
complexity with no assembly-cost win, which is this tier's own gate for doing the
work.
"""

from typing import Any, Optional, Tuple

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.backends.diagonal.diagonal_jacobian import (
    DiagonalJacobian,
)
from pymrst.modules.ad_core.backends.diagonal.generic_ad import GenericAD
from pymrst.modules.ad_core.backends.diagonal.utils.diag_mult import diag_mult
from pymrst.modules.ad_core.backends.diagonal.utils.get_sparse_arguments import get_sparse_arguments
from pymrst.modules.ad_core.backends.diagonal.utils.matrix_dims import matrix_dims


def diag_product_mult(v1, v2, x, y, d1=None, d2=None):
    """MRST ``diagProductMult``: ``diag(v2)x + diag(v1)y``."""
    return diag_mult(v2, x, d2) + diag_mult(v1, y, d1), d1, d2


def double_to_generic_ad(v, sample):
    """Convert numeric ``v`` to a constant ``GenericAD`` shaped like sample."""
    values = np.atleast_1d(np.asarray(v, dtype=float)).ravel()
    if not isinstance(sample, GenericAD):
        return np.asarray(v, dtype=float)
    blocks = [
        type(block).zeros(values.size, block.shape[1])
        for block in sample.jac
    ]
    return GenericAD(values, blocks)


def get_sparse_blocks(*args, **kwargs):
    """MRST ``getSparseBlocks`` forwards to ``getSparseArguments``."""
    return get_sparse_arguments(*args, **kwargs)


def increment_subset(x, subset, v):
    """Increment selected rows while preserving AD derivatives."""
    if isinstance(x, GenericAD):
        out = x.copy()
        out[subset] = out[subset] + v
        return out
    out = np.array(x, dtype=np.result_type(x, v), copy=True)
    out[subset] += v
    return out


diagMult = diag_mult
diagProductMult = diag_product_mult
double2GenericAD = double_to_generic_ad
getSparseArguments = get_sparse_arguments
getSparseBlocks = get_sparse_blocks
incrementSubset = increment_subset
matrixDims = matrix_dims

__all__ = [
    "diag_mult", "diagMult",
    "diag_product_mult", "diagProductMult",
    "double_to_generic_ad", "double2GenericAD",
    "get_sparse_arguments", "getSparseArguments",
    "get_sparse_blocks", "getSparseBlocks",
    "increment_subset", "incrementSubset",
    "matrix_dims", "matrixDims",
]

from pymrst.modules.ad_core.backends.diagonal.utils.init_variables_ad_diagonal import (
    init_variables_ad_diagonal,
    initVariablesAD_diagonal,
)
from pymrst.modules.ad_core.backends.diagonal.utils.init_variables_ad_diagonal_row_major import (
    init_variables_ad_diagonal_row_major,
    initVariablesAD_diagonalRowMajor,
)
from pymrst.modules.ad_core.backends.diagonal.utils.init_variables_ad_one_block import (
    init_variables_one_block,
    initVariablesAD_oneBlock,
)

__all__ += [
    "init_variables_ad_diagonal", "initVariablesAD_diagonal",
    "init_variables_ad_diagonal_row_major", "initVariablesAD_diagonalRowMajor",
    "init_variables_one_block", "initVariablesAD_oneBlock",
]


if __name__ == "__main__":
    def test_matrix_dims_reads_shape():
        j = DiagonalJacobian.identity(4)
        assert matrix_dims(j) == (4, 4)
        assert matrix_dims(sp.csr_array((2, 5))) == (2, 5)
        assert matrix_dims(np.zeros((3, 7))) == (3, 7)
        print("ok test_matrix_dims_reads_shape")

    def test_get_sparse_arguments_triplet_rebuilds_matrix():
        dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
        j = DiagonalJacobian.from_sparse(dense)
        i, jj, v = get_sparse_arguments(j)
        rebuilt = sp.csr_array((v, (i, jj)), shape=j.shape).toarray()
        assert np.allclose(rebuilt, dense)
        print("ok test_get_sparse_arguments_triplet_rebuilds_matrix")

    def test_get_sparse_arguments_offsets_place_a_block():
        """Offsets drop a 2x3 block at (row 4, col 1) of a larger system."""
        dense = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
        i, jj, v = get_sparse_arguments(sp.csr_array(dense), ioffset=4, joffset=1)
        big = sp.csr_array((v, (i, jj)), shape=(8, 6)).toarray()
        assert np.allclose(big[4:6, 1:4], dense)
        assert big.sum() == dense.sum()          # nothing placed elsewhere
        print("ok test_get_sparse_arguments_offsets_place_a_block")

    def test_diag_mult_matches_dense_row_scaling():
        m = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        v = np.array([10.0, 0.0, 2.0])
        got = diag_mult(v, m).toarray()
        assert np.allclose(got, np.diag(v) @ m)
        print("ok test_diag_mult_matches_dense_row_scaling")

    def test_diag_mult_fast_paths():
        m = sp.csr_array(np.array([[1.0, 2.0], [3.0, 4.0]]))
        # all-zero v -> empty
        assert diag_mult(np.zeros(2), m).nnz == 0
        # scalar v -> v*m
        assert np.allclose(diag_mult(np.array([3.0]), m).toarray(),
                           3.0 * m.toarray())
        # zero matrix -> unchanged (still zero)
        z = sp.csr_array((2, 2))
        assert diag_mult(np.array([1.0, 2.0]), z).nnz == 0
        print("ok test_diag_mult_fast_paths")

    test_matrix_dims_reads_shape()
    test_get_sparse_arguments_triplet_rebuilds_matrix()
    test_get_sparse_arguments_offsets_place_a_block()
    test_diag_mult_matches_dense_row_scaling()
    test_diag_mult_fast_paths()
    print("\nAll tests passed!")
