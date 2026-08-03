"""AD-aware table interpolation, ported from MRST ``utils/interpTable.m``.

Linear interpolation with linear extrapolation at both ends, transparent to
ADI values. This lives in ``pymrst/utils`` because MRST puts ``interpTable``
in its core ``utils/``, not in ``ad-props``.
"""

from typing import Tuple, Union

import numpy as np

from pymrst.utils.adi import ADI, _diag_mul, value

ArrayOrADI = Union[np.ndarray, ADI]


def _interp1(x_tab: np.ndarray, f_tab: np.ndarray,
             x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Linear interp with linear extrapolation; returns (f, df/dx)."""
    idx = np.clip(np.searchsorted(x_tab, x, side="right") - 1,
                  0, x_tab.size - 2)
    x0, x1 = x_tab[idx], x_tab[idx + 1]
    f0, f1 = f_tab[idx], f_tab[idx + 1]
    slope = (f1 - f0) / (x1 - x0)
    return f0 + slope * (x - x0), slope


def interp_table(x_tab: np.ndarray, f_tab: np.ndarray,
                 x: ArrayOrADI, spline: bool = False,
                 return_derivative: bool = False) -> ArrayOrADI:
    """MRST ``interpTable`` with linear/spline extrapolation.

    ``return_derivative=True`` represents MATLAB's optional second output.
    """
    xv = value(x)
    x_tab = np.asarray(x_tab, float).ravel()
    f_tab = np.asarray(f_tab, float).ravel()
    if x_tab.size != f_tab.size or x_tab.size < 2:
        raise ValueError("x_tab and f_tab must contain at least two matching values")
    if np.any(np.diff(x_tab) < 0):
        x_tab, f_tab = x_tab[::-1], f_tab[::-1]
    if np.any(np.diff(x_tab) == 0):
        raise ValueError("x_tab values must be strictly monotone")
    if spline:
        from scipy.interpolate import CubicSpline
        pp = CubicSpline(x_tab, f_tab)
        f, df = pp(xv), pp.derivative()(xv)
    else:
        f, df = _interp1(x_tab, f_tab, xv)
    if isinstance(x, ADI):
        result = ADI(f, [_diag_mul(df, j) for j in x.jac])
    else:
        result = f
    return (result, df) if return_derivative else result


interpTable = interp_table

# `_interp1` returns the slope alongside the value, which the PVT
# interpolators need to build their own derivatives
__all__ = ["interp_table", "interpTable", "_interp1", "ArrayOrADI"]


if __name__ == "__main__":

    def test_interpolates_between_nodes() -> None:
        x = np.array([0.0, 1.0, 2.0])
        f = np.array([0.0, 10.0, 30.0])
        assert np.allclose(interp_table(x, f, np.array([0.5, 1.5])), [5.0, 20.0])
        print("ok test_interpolates_between_nodes")

    def test_extrapolates_linearly_at_both_ends() -> None:
        x = np.array([0.0, 1.0, 2.0])
        f = np.array([0.0, 10.0, 30.0])
        assert np.allclose(interp_table(x, f, np.array([-1.0])), [-10.0])
        assert np.allclose(interp_table(x, f, np.array([3.0])), [50.0])
        print("ok test_extrapolates_linearly_at_both_ends")

    def test_reproduces_the_table_at_its_nodes() -> None:
        x = np.array([0.0, 1.0, 2.5, 4.0])
        f = np.array([1.0, 3.0, -2.0, 7.0])
        assert np.allclose(interp_table(x, f, x), f)
        print("ok test_reproduces_the_table_at_its_nodes")

    def test_carries_derivatives_through_adi() -> None:
        from pymrst.utils.adi import init_variables_adi, value as adi_value
        x = np.array([0.0, 1.0, 2.0])
        f = np.array([0.0, 10.0, 30.0])
        (p,) = init_variables_adi(np.array([0.5, 1.5]))
        out = interp_table(x, f, p)
        assert isinstance(out, ADI)
        assert np.allclose(adi_value(out), [5.0, 20.0])
        # slopes are 10 on the first interval and 20 on the second
        assert np.allclose(out.jac[0].toarray().diagonal(), [10.0, 20.0])
        print("ok test_carries_derivatives_through_adi")

    test_interpolates_between_nodes()
    test_extrapolates_linearly_at_both_ends()
    test_reproduces_the_table_at_its_nodes()
    test_carries_derivatives_through_adi()
    print("\nAll tests passed!")
