"""Diagonal AD backend."""

from pymrst.modules.ad_core.backends.diagonal.diagonal_auto_diff_backend import (
    DiagonalAutoDiffBackend,
)
from pymrst.modules.ad_core.backends.diagonal.diagonal_jacobian import (
    DiagonalJacobian,
    diagonal_hstack,
)
from pymrst.modules.ad_core.backends.diagonal.row_major_diagonal_jacobian import (
    RowMajorDiagonalJacobian,
)
from pymrst.modules.ad_core.backends.diagonal.generic_ad import (
    GenericAD,
    combine_diagonal_equations,
    diagonal_ad_matmul,
    diagonal_sum,
    init_variables_diagonal,
)
from pymrst.modules.ad_core.backends.diagonal.operators import (
    discrete_divergence,
    face_average,
    setup_diagonal_operator_closures,
    single_point_upwind,
    sparse_from_diagonal_ad,
    two_point_gradient,
)

__all__ = [
    "DiagonalAutoDiffBackend",
    "DiagonalJacobian",
    "GenericAD",
    "RowMajorDiagonalJacobian",
    "combine_diagonal_equations",
    "diagonal_ad_matmul",
    "diagonal_hstack",
    "diagonal_sum",
    "discrete_divergence",
    "face_average",
    "init_variables_diagonal",
    "setup_diagonal_operator_closures",
    "single_point_upwind",
    "sparse_from_diagonal_ad",
    "two_point_gradient",
]
