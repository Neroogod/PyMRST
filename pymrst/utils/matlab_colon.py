"""MATLAB's colon operator, `a:d:b`, value for value.

`np.arange(a, b + d, d)` computes `a + k*d`; MATLAB fills the vector from
both ends, `a + k*d` for the first half and `c - k*d` for the second (`c`
being `b` when `a + n*d` is within round-off of it), with the exact midpoint
`(a + c)/2` when the count is odd. The two differ in the last bit for many
elements (2156 of `0:0.0001:1`), which changes the outcome of exact tests
such as `kr == 0` on a sampled saturation range.
"""

from __future__ import annotations

import math

import numpy as np


def matlab_colon(a: float, d: float, b: float) -> np.ndarray:
    """MATLAB `a:d:b`."""
    a, d, b = float(a), float(d), float(b)
    if d == 0 or (d > 0 and a > b) or (d < 0 and a < b) or math.isnan(a + d + b):
        return np.zeros(0)
    tol = 2.0*np.finfo(float).eps*max(abs(a), abs(b))
    sig = math.copysign(1.0, d)
    n = math.floor((b - a)/d)
    if sig*(a + (n + 1)*d - b) <= tol:          # one more element is within round-off
        n += 1
    c = a + n*d
    if abs(c - b) < tol:
        c = b
    v = np.zeros(n + 1)
    k = np.arange(0, n//2 + 1)
    v[k] = a + k*d
    v[n - k] = c - k*d
    if n % 2 == 0:
        v[n//2] = (a + c)/2
    return v


__all__ = ["matlab_colon"]


if __name__ == "__main__":
    def test_the_ends_and_the_count():
        v = matlab_colon(0, 0.1, 1)
        assert v.size == 11 and v[0] == 0 and v[-1] == 1
        assert matlab_colon(0, 0.25, 0.9).tolist() == [0, 0.25, 0.5, 0.75]
        assert matlab_colon(1, 1, 0).size == 0
        print("ok test_the_ends_and_the_count")

    test_the_ends_and_the_count()
    print("\nAll tests passed!")
