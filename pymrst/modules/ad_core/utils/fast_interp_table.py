"""Fast 1D table lookup with linear extrapolation (MRST `fastInterpTable`).

A thin wrapper over piecewise-linear interpolation that, unlike a clamped
lookup, *extrapolates linearly* past the ends of the table - matching MATLAB's
`griddedInterpolant(X, Y, 'linear', 'linear')`. Reservoir property tables (PVT,
relperm) are queried at states that occasionally fall just outside the tabulated
range, and a linear continuation is the sane behaviour there rather than a flat
clamp or a NaN.
"""

import numpy as np
from scipy.interpolate import interp1d


def fast_interp_table(x: np.ndarray, y: np.ndarray, xi: np.ndarray) -> np.ndarray:
    """Linearly interpolate `(x, y)` at `xi`, extrapolating linearly outside.

    `x` must be sorted ascending. Returns an empty array for empty `xi`,
    mirroring MRST's early return.
    """
    xi = np.asarray(xi, dtype=float)
    if xi.size == 0:
        return np.array([])
    f = interp1d(np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                 kind="linear", fill_value="extrapolate", assume_sorted=True)
    return f(xi)


#: MRST's own spelling. This port names functions in snake_case; the
#: alias keeps the MATLAB name callable, which is the convention the
#: rest of the package already follows.
fastInterpTable = fast_interp_table

if __name__ == "__main__":
    def test_interior_matches_linear_interpolation():
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 10.0, 20.0, 30.0])
        yi = fast_interp_table(x, y, np.array([0.5, 1.5, 2.25]))
        assert np.allclose(yi, [5.0, 15.0, 22.5]), yi
        print("ok test_interior_matches_linear_interpolation")

    def test_extrapolates_linearly_past_both_ends():
        """Past the ends the slope of the nearest segment continues - not a
        clamp. Left slope 10, right slope 10, so -1 -> -10 and 4 -> 40."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = np.array([0.0, 10.0, 20.0, 30.0])
        yi = fast_interp_table(x, y, np.array([-1.0, 4.0]))
        assert np.allclose(yi, [-10.0, 40.0]), yi
        print("ok test_extrapolates_linearly_past_both_ends")

    def test_empty_query_returns_empty():
        yi = fast_interp_table(np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                               np.array([]))
        assert yi.size == 0
        print("ok test_empty_query_returns_empty")

    test_interior_matches_linear_interpolation()
    test_extrapolates_linearly_past_both_ends()
    test_empty_query_returns_empty()
    print("\nAll tests passed!")
