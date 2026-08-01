"""Geometrically graded spacing (MRST ``utils/geomspace.m``).

`n` points from `a` to `b` whose *increments* form a geometric sequence
starting at `x0`. Near-well grids use it: fine cells at the well, coarsening
outward, with the ratio chosen so the increments happen to sum to `b - a`.

That ratio is not free. Given the endpoints, the count and the first
increment, it is the unique root greater than one of

    q^(n-1) - f q + (f - 1) = 0,    f = (b - a) / x0

so `geomspace` solves a polynomial rather than picking a stretch factor. The
consequence is a constraint that surprises people: **a sequence exists only
when `f > n - 1`**, that is, when the total distance exceeds what `n - 1`
steps of the *initial* size would cover. Ask for a first step too large and
there is no answer, and MRST raises rather than returning something close.
"""

import numpy as np


def geomspace(a: float, b: float, n: int, x0: float) -> np.ndarray:
    """``n`` points from ``a`` to ``b`` with geometrically growing steps."""
    if n < 2:
        raise ValueError(f"a geometric sequence needs at least two points, "
                         f"got {n}")
    f = (b - a) / x0
    if not f > n - 1:
        raise ValueError(
            f"no geometric sequence exists with these parameters: the span "
            f"({b - a:g}) must exceed {n - 1} initial steps of {x0:g}")

    # q^(n-1) - f q + (f - 1) = 0, in numpy's descending-power convention.
    coefficients = np.zeros(n)
    coefficients[0] = 1.0
    coefficients[-2] = -f
    coefficients[-1] = f - 1.0
    roots = np.roots(coefficients)
    candidates = roots[(np.abs(roots.imag) == 0) & (roots.real > 1)]
    if candidates.size != 1:
        raise RuntimeError(
            f"expected one growth ratio above 1, found {candidates.size}")
    q = float(candidates[0].real)

    return np.cumsum(np.concatenate([[a], x0 * q ** np.arange(n - 1)]))


if __name__ == "__main__":

    def test_it_spans_the_requested_interval() -> None:
        x = geomspace(0.0, 100.0, 10, 1.0)
        assert np.isclose(x[0], 0.0) and np.isclose(x[-1], 100.0), (x[0], x[-1])
        print("ok test_it_spans_the_requested_interval")

    def test_it_returns_the_requested_number_of_points() -> None:
        assert geomspace(0.0, 100.0, 10, 1.0).size == 10
        print("ok test_it_returns_the_requested_number_of_points")

    def test_the_first_step_is_the_one_asked_for() -> None:
        x = geomspace(0.0, 100.0, 10, 1.0)
        assert np.isclose(x[1] - x[0], 1.0), x[1] - x[0]
        print("ok test_the_first_step_is_the_one_asked_for")

    def test_the_increments_form_a_geometric_sequence() -> None:
        x = geomspace(0.0, 100.0, 10, 1.0)
        steps = np.diff(x)
        ratios = steps[1:] / steps[:-1]
        assert np.allclose(ratios, ratios[0]), ratios
        assert ratios[0] > 1.0, ratios[0]
        print(f"ok test_the_increments_form_a_geometric_sequence "
              f"(ratio {ratios[0]:.4f})")

    def test_the_points_increase() -> None:
        assert np.all(np.diff(geomspace(0.0, 50.0, 8, 0.5)) > 0)
        print("ok test_the_points_increase")

    def test_an_impossible_first_step_is_rejected() -> None:
        """10 steps of 20 cannot cover 100; there is no such sequence."""
        try:
            geomspace(0.0, 100.0, 10, 20.0)
        except ValueError as error:
            assert "no geometric sequence exists" in str(error)
            print("ok test_an_impossible_first_step_is_rejected")
            return
        raise AssertionError("an over-large first step must be rejected")

    def test_the_boundary_case_is_excluded() -> None:
        """f == n-1 means uniform spacing, and the polynomial degenerates."""
        try:
            geomspace(0.0, 9.0, 10, 1.0)          # f = 9 = n - 1
        except ValueError:
            print("ok test_the_boundary_case_is_excluded")
            return
        raise AssertionError("f == n - 1 must be rejected")

    def test_too_few_points_is_rejected() -> None:
        try:
            geomspace(0.0, 1.0, 1, 0.1)
        except ValueError as error:
            assert "at least two points" in str(error)
            print("ok test_too_few_points_is_rejected")
            return
        raise AssertionError("n < 2 must be rejected")

    test_it_spans_the_requested_interval()
    test_it_returns_the_requested_number_of_points()
    test_the_first_step_is_the_one_asked_for()
    test_the_increments_form_a_geometric_sequence()
    test_the_points_increase()
    test_an_impossible_first_step_is_rejected()
    test_the_boundary_case_is_excluded()
    test_too_few_points_is_rejected()
    print("\nAll tests passed!")
