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
                 x: ArrayOrADI) -> ArrayOrADI:
    """MRST interpTable: linear with extrapolation, ADI-transparent."""
    xv = value(x)
    f, df = _interp1(np.asarray(x_tab, float), np.asarray(f_tab, float), xv)
    if isinstance(x, ADI):
        return ADI(f, [_diag_mul(df, j) for j in x.jac])
    return f


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
