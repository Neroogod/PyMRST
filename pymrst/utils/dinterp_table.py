"""Derivative of `interp_table` (MRST ``utils/dinterpTable.m``).

A thin selector: piecewise-linear by default, delegating to `dinterpq1`, or the
derivative of a cubic spline when asked.

The spline branch differentiates the *spline*, not the data - it builds the
interpolating cubic and returns its analytic derivative, which is continuous.
The default branch returns a piecewise-constant slope. So the `spline` flag
changes the answer's smoothness, not merely its accuracy, and a solver relying
on a continuous derivative needs it set.
"""

import numpy as np

from pymrst.utils.dinterpq1 import dinterpq1


def dinterp_table(x, y, xi, spline: bool = False):
    """Slope of the interpolant of ``(x, y)`` at ``xi``."""
    if not spline:
        return dinterpq1(x, y, xi)

    from scipy.interpolate import CubicSpline
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if np.any(np.diff(x) < 0):
        x, y = x[::-1], y[::-1]
    return CubicSpline(x, y).derivative()(np.asarray(xi, dtype=float))


dinterpTable = dinterp_table


if __name__ == "__main__":

    X = np.array([0.0, 1.0, 3.0, 6.0])
    Y = np.array([0.0, 2.0, 2.0, 8.0])

    def test_the_default_is_the_piecewise_linear_slope() -> None:
        probe = np.array([0.5, 2.0, 4.5])
        assert np.allclose(dinterp_table(X, Y, probe), dinterpq1(X, Y, probe))
        print("ok test_the_default_is_the_piecewise_linear_slope")

    def test_the_spline_branch_is_continuous_where_the_default_jumps() -> None:
        """The flag changes smoothness, not just accuracy."""
        near = np.array([0.999, 1.001])
        linear = dinterp_table(X, Y, near)
        smooth = dinterp_table(X, Y, near, spline=True)
        assert abs(linear[1] - linear[0]) > 1.0, linear
        assert abs(smooth[1] - smooth[0]) < 1e-2, smooth
        print("ok test_the_spline_branch_is_continuous_where_the_default_jumps")

    def test_the_spline_derivative_matches_a_finite_difference() -> None:
        from scipy.interpolate import CubicSpline
        spline = CubicSpline(X, Y)
        probe = np.array([0.4, 2.2, 4.8])
        step = 1e-6
        numeric = (spline(probe + step) - spline(probe - step)) / (2 * step)
        assert np.allclose(dinterp_table(X, Y, probe, spline=True), numeric,
                           atol=1e-6)
        print("ok test_the_spline_derivative_matches_a_finite_difference")

    def test_a_linear_table_gives_the_same_answer_either_way() -> None:
        """Where the data is linear, the spline is too."""
        x = np.array([0.0, 1.0, 2.0, 3.0])
        y = 3.0 * x + 1.0
        probe = np.array([0.5, 1.5, 2.5])
        assert np.allclose(dinterp_table(x, y, probe), 3.0)
        assert np.allclose(dinterp_table(x, y, probe, spline=True), 3.0)
        print("ok test_a_linear_table_gives_the_same_answer_either_way")

    test_the_default_is_the_piecewise_linear_slope()
    test_the_spline_branch_is_continuous_where_the_default_jumps()
    test_the_spline_derivative_matches_a_finite_difference()
    test_a_linear_table_gives_the_same_answer_either_way()
    print("\nAll tests passed!")
