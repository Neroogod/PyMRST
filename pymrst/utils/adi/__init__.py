"""Automatic differentiation helpers, ported from MRST `utils/ADI.m`."""

from pymrst.utils.adi.adi import ADI, _diag_mul, ad_from_blocks, is_ad
from pymrst.utils.adi.adi2 import (
    ADI2,
    ad2_atan2,
    ad2_cos,
    ad2_linsolve,
    ad2_matmul,
    ad2_scatter,
    ad2_sin,
    ad2_sum,
    ad2_vertcat,
    hessian_ad2,
    init_variables_adi2,
    value_jac_hess,
)
from pymrst.utils.adi.init_variables_adi import init_variables_adi
from pymrst.utils.adi.operators import (
    ad_abs,
    ad_exp,
    ad_log,
    ad_matmul,
    ad_max,
    ad_min,
    ad_sum,
    combine_equations,
    vertcat_adi,
)
from pymrst.utils.adi.value import value

__all__ = [
    "ADI",
    "ADI2",
    "_diag_mul",
    "ad_from_blocks",
    "is_ad",
    "ad2_matmul",
    "ad2_atan2",
    "ad2_cos",
    "ad2_linsolve",
    "ad2_scatter",
    "ad2_sin",
    "ad2_sum",
    "ad2_vertcat",
    "ad_abs",
    "ad_exp",
    "ad_log",
    "ad_matmul",
    "ad_max",
    "ad_min",
    "ad_sum",
    "combine_equations",
    "hessian_ad2",
    "init_variables_adi",
    "init_variables_adi2",
    "value",
    "value_jac_hess",
    "vertcat_adi",
]
