"""Automatic-differentiation backends.

MRST exposes concrete backend objects such as `SparseAutoDiffBackend` and
`DiagonalAutoDiffBackend`. PyMRST implements sparse SciPy Jacobians and a
pure-Python diagonal backend with explicit sparse handoff for linear solves.
"""

from pymrst.modules.ad_core.backends.auto_diff_backend import AutoDiffBackend
from pymrst.modules.ad_core.backends.diagonal.diagonal_auto_diff_backend import (
    DiagonalAutoDiffBackend,
)
from pymrst.modules.ad_core.backends.diagonal.diagonal_jacobian import DiagonalJacobian
from pymrst.modules.ad_core.backends.diagonal.generic_ad import (
    GenericAD,
    combine_diagonal_equations,
    init_variables_diagonal,
)
from pymrst.modules.ad_core.backends.diagonal.row_major_diagonal_jacobian import (
    RowMajorDiagonalJacobian,
)
from pymrst.modules.ad_core.backends.sparse.sparse_auto_diff_backend import (
    SparseAutoDiffBackend,
)

__all__ = [
    "AutoDiffBackend",
    "DiagonalAutoDiffBackend",
    "DiagonalJacobian",
    "GenericAD",
    "RowMajorDiagonalJacobian",
    "SparseAutoDiffBackend",
    "combine_diagonal_equations",
    "init_variables_diagonal",
]
