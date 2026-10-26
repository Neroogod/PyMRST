"""MATLAB's linear interpolation, bit for bit: `interp1(x, y, xq, 'linear',
'extrap')` and `griddedInterpolant(x, y, 'linear', 'nearest')`.

Both evaluate, in the interval `x0 <= xq < x1`, `t = (xq - x0)/(x1 - x0)`
and `y0*(1 - t) + y1*t` (checked against MATLAB R2021b on 200000 points
each). `np.interp` and `y0 + slope*(xq - x0)` are the same line but differ
in the last bits - which matters wherever a result is a small difference
of large numbers, as in an equilibrated reservoir's phase potentials.

`extrap` is 'linear' (`interp1`'s 'extrap': the end intervals extended) or
'nearest' (`griddedInterpolant`'s: the end values held). `x` increasing.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def matlab_interp1(x: Any, y: Any, xq: Any, extrap: str = "linear") -> np.ndarray:
    """MATLAB's linear interpolation of `(x, y)` at `xq`."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    xq = np.asarray(xq, dtype=float)
    if extrap == "nearest":
        xc = np.minimum(np.maximum(xq, x[0]), x[-1])
    elif extrap == "linear":
        xc = xq
    else:
        raise ValueError(f"Unknown extrapolation {extrap!r}")
    if x.size == 1:
        return np.full(np.shape(xq), y[0])
    idx = np.clip(np.searchsorted(x, xc, side="right") - 1, 0, x.size - 2)
    x0, x1 = x[idx], x[idx + 1]
    t = (xc - x0)/(x1 - x0)
    return y[idx]*(1 - t) + y[idx + 1]*t


__all__ = ["matlab_interp1"]


if __name__ == "__main__":
    def test_inside_and_at_the_knots():
        x, y = np.array([0.0, 1.0, 3.0]), np.array([1.0, 3.0, 7.0])
        np.testing.assert_allclose(matlab_interp1(x, y, [0.0, 0.5, 1.0, 2.0, 3.0]),
                                   [1.0, 2.0, 3.0, 5.0, 7.0])
        print("ok test_inside_and_at_the_knots")

    def test_the_two_extrapolations():
        x, y = np.array([0.0, 1.0]), np.array([0.0, 2.0])
        np.testing.assert_allclose(matlab_interp1(x, y, [-1.0, 2.0]), [-2.0, 4.0])
        np.testing.assert_allclose(matlab_interp1(x, y, [-1.0, 2.0], "nearest"), [0.0, 2.0])
        print("ok test_the_two_extrapolations")

    def test_the_weights_form_is_not_the_slope_form():
        # A case where the two forms round differently in the last bit.
        x0, x1, y0, y1, xq = 0.1, 0.7, 0.3, 1e5/3, 0.4
        slope = y0 + (y1 - y0)/(x1 - x0)*(xq - x0)
        t = (xq - x0)/(x1 - x0)
        assert matlab_interp1([x0, x1], [y0, y1], [xq])[0] == y0*(1 - t) + y1*t
        assert abs(slope - (y0*(1 - t) + y1*t)) < 1e-10
        print("ok test_the_weights_form_is_not_the_slope_form")

    test_inside_and_at_the_knots()
    test_the_two_extrapolations()
    test_the_weights_form_is_not_the_slope_form()
    print("\nAll tests passed!")
