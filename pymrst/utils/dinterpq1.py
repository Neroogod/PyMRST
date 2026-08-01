"""Derivative of piecewise-linear interpolation (MRST ``utils/dinterpq1.m``).

The slope of `interp_table`'s interpolant at each query point. Because the
interpolant is piecewise linear its derivative is piecewise *constant*, so this
returns the slope of the interval each point falls in - not a smoothed or
averaged value.

Two consequences follow, and both matter to a Newton iteration built on these
tables:

- **At a breakpoint the derivative is one-sided.** There is no derivative
  there in the strict sense, and any choice is arbitrary; MRST takes the
  interval to the right, and so does this.
- **Outside the table the derivative is the end interval's slope**, not zero.
  That matches `interp_table`, which extrapolates linearly rather than
  clamping flat, so the value and its derivative stay consistent all the way
  out. Worth stating because the other table mechanism in this codebase,
  `extend_tab`, *does* flatten - it appends rows to the table - and the two are
  easy to confuse.
"""

import numpy as np


def dinterpq1(x, y, xi):
    """Slope of the piecewise-linear interpolant of ``(x, y)`` at ``xi``."""
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size < 2:
        raise ValueError("at least two table points are needed for a slope")
    if x.size != y.size:
        raise ValueError(f"x and y must match in size, got {x.size} and {y.size}")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("the table may not contain infinities")

    query = np.asarray(xi, dtype=float)
    if not np.all(np.isfinite(query)):
        raise ValueError("the query points may not contain infinities")

    if np.any(np.diff(x) < 0):
        x, y = x[::-1], y[::-1]

    slopes = np.diff(y) / np.diff(x)
    # Repeat the end slopes so a point outside the table extrapolates.
    slopes = np.concatenate([slopes[:1], slopes, slopes[-1:]])

    # Which interval each point falls in, counting the two outer ones.
    bins = np.searchsorted(x, query.ravel(), side="right")
    return slopes[bins].reshape(query.shape)


if __name__ == "__main__":

    X = np.array([0.0, 1.0, 3.0, 6.0])
    Y = np.array([0.0, 2.0, 2.0, 8.0])          # slopes 2, 0, 2

    def test_the_slope_of_each_interval_is_returned() -> None:
        got = dinterpq1(X, Y, np.array([0.5, 2.0, 4.5]))
        assert np.allclose(got, [2.0, 0.0, 2.0]), got
        print("ok test_the_slope_of_each_interval_is_returned")

    def test_it_matches_a_finite_difference_of_interp_table() -> None:
        """The property that makes it the derivative of the interpolant."""
        from pymrst.utils.interp_table import interp_table
        probe = np.array([0.3, 1.7, 2.5, 4.0, 5.5])
        step = 1e-7
        numeric = ((interp_table(X, Y, probe + step)
                    - interp_table(X, Y, probe - step)) / (2 * step))
        assert np.allclose(dinterpq1(X, Y, probe), numeric, atol=1e-6)
        print("ok test_it_matches_a_finite_difference_of_interp_table")

    def test_the_derivative_is_piecewise_constant() -> None:
        inside = np.linspace(1.05, 2.95, 20)
        assert np.allclose(dinterpq1(X, Y, inside), 0.0)
        print("ok test_the_derivative_is_piecewise_constant")

    def test_at_a_breakpoint_it_takes_the_interval_to_the_right() -> None:
        """Arbitrary but pinned: there is no two-sided derivative there."""
        assert np.isclose(dinterpq1(X, Y, np.array([1.0]))[0], 0.0)
        print("ok test_at_a_breakpoint_it_takes_the_interval_to_the_right")

    def test_outside_the_table_the_end_slope_is_used() -> None:
        got = dinterpq1(X, Y, np.array([-5.0, 99.0]))
        assert np.allclose(got, [2.0, 2.0]), got
        print("ok test_outside_the_table_the_end_slope_is_used")

    def test_it_stays_consistent_with_interp_table_outside_the_range() -> None:
        """Both extrapolate linearly, so the slope still describes the value."""
        from pymrst.utils.interp_table import interp_table
        far = np.array([20.0, 21.0])
        values = interp_table(X, Y, far)
        assert np.isclose(values[1] - values[0], dinterpq1(X, Y, far)[0]), \
            (values, dinterpq1(X, Y, far))
        print("ok test_it_stays_consistent_with_interp_table_outside_the_range")

    def test_a_decreasing_table_is_reordered() -> None:
        got = dinterpq1(X[::-1], Y[::-1], np.array([0.5]))
        assert np.allclose(got, 2.0), got
        print("ok test_a_decreasing_table_is_reordered")

    def test_the_query_shape_is_preserved() -> None:
        query = np.array([[0.5, 2.0], [4.5, 0.5]])
        assert dinterpq1(X, Y, query).shape == query.shape
        print("ok test_the_query_shape_is_preserved")

    def test_a_one_point_table_is_rejected() -> None:
        try:
            dinterpq1([1.0], [2.0], [1.0])
        except ValueError as error:
            assert "two table points" in str(error)
            print("ok test_a_one_point_table_is_rejected")
            return
        raise AssertionError("a slope needs two points")

    def test_mismatched_lengths_are_rejected() -> None:
        try:
            dinterpq1([0.0, 1.0], [0.0, 1.0, 2.0], [0.5])
        except ValueError as error:
            assert "match in size" in str(error)
            print("ok test_mismatched_lengths_are_rejected")
            return
        raise AssertionError("mismatched table columns must be rejected")

    test_the_slope_of_each_interval_is_returned()
    test_it_matches_a_finite_difference_of_interp_table()
    test_the_derivative_is_piecewise_constant()
    test_at_a_breakpoint_it_takes_the_interval_to_the_right()
    test_outside_the_table_the_end_slope_is_used()
    test_it_stays_consistent_with_interp_table_outside_the_range()
    test_a_decreasing_table_is_reordered()
    test_the_query_shape_is_preserved()
    test_a_one_point_table_is_rejected()
    test_mismatched_lengths_are_rejected()
    print("\nAll tests passed!")
