"""MRST ``tableByPressureLinearAD.m`` pressure-table interpolation."""

import numpy as np

from pymrst.utils.interp_table import _interp1
from pymrst.utils.adi import ADI, _diag_mul, value


def table_by_pressure_linear_ad(x_tab: np.ndarray, f_tab: np.ndarray, x):
    """Linearly interpolate/extrapolate pressure tables for numeric or ADI input."""
    xv = value(x)
    f, df = _interp1(np.asarray(x_tab, float), np.asarray(f_tab, float), xv)
    if isinstance(x, ADI):
        return ADI(f, [_diag_mul(df, j) for j in x.jac])
    return f

__all__ = ["table_by_pressure_linear_ad"]
