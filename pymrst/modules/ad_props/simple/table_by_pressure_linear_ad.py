"""MRST ``ad-props/simple/tableByPressureLinearAD.m``.

Functions of pressure from tables: for each table of values given at the
pressures `pRef`, a function `f(p, ...)` interpolating it linearly with
`interpTable` (linear extrapolation outside the table).

`table_by_pressure_linear_ad_ad` is PyMRST's earlier version, which
interpolates one table at given pressures directly.
"""

from __future__ import annotations

from typing import Any, Callable, Tuple

import numpy as np

from pymrst.utils.adi import _diag_mul, ad_from_blocks, is_ad, value
from pymrst.utils.interp_table import _interp1, interp_table


def table_by_pressure_linear_ad(pRef: Any, *tables: Any) -> Tuple[Callable[..., Any], ...]:
    """MRST `[f1, f2, ...] = tableByPressureLinearAD(pRef, v1, v2, ...)`."""
    pRef = np.asarray(pRef, dtype=float).ravel()
    out = []
    for v in tables:
        v = np.asarray(v, dtype=float).reshape(-1)
        assert v.size == pRef.size, "Inconsistent table dimensions"
        out.append(lambda p, *varargin, v=v: interp_table(pRef, v, p))
    return tuple(out)


def table_by_pressure_linear_ad_ad(x_tab: np.ndarray, f_tab: np.ndarray, x):
    """PyMRST's earlier version: interpolate one table at `x`."""
    xv = value(x)
    f, df = _interp1(np.asarray(x_tab, float), np.asarray(f_tab, float), xv)
    if is_ad(x):
        return ad_from_blocks(f, [_diag_mul(df, j) for j in x.jac])
    return f


tableByPressureLinearAD = table_by_pressure_linear_ad

__all__ = ["tableByPressureLinearAD", "table_by_pressure_linear_ad",
           "table_by_pressure_linear_ad_ad"]


if __name__ == "__main__":
    def test_one_function_per_table():
        b, mu = table_by_pressure_linear_ad([1.0, 2.0, 4.0], [1.0, 0.5, 0.25], [3.0, 4.0, 4.0])
        np.testing.assert_allclose(b(np.array([1.5, 3.0, 5.0])), [0.75, 0.375, 0.125])
        np.testing.assert_allclose(mu(np.array([1.5]), "ignored"), [3.5])
        print("ok test_one_function_per_table")

    def test_inconsistent_tables_are_refused():
        try:
            table_by_pressure_linear_ad([1.0, 2.0], [1.0])
        except AssertionError:
            pass
        else:
            raise AssertionError("expected AssertionError")
        print("ok test_inconsistent_tables_are_refused")

    test_one_function_per_table()
    test_inconsistent_tables_are_refused()
    print("\nAll tests passed!")
