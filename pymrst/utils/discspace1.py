"""Graded one-dimensional spacing (MRST ``utils/discspace1.m``).

`N` points from `d1` to `d2`, with local refinement given as `(position,
factor)` rows: a factor above 1 packs points in near that position, below 1
spreads them out.

Two schemes, and they are not variations of one idea:

- ``linear`` (and any other name `numpy.interp`-style interpolation accepts)
  builds a cumulative density from the factors, then places `N` points at
  equal *density* intervals. The refinement is smooth: a factor at one
  position influences the spacing either side of it.
- ``constant`` treats each refinement interval separately, giving each a point
  count proportional to its factor and spacing them uniformly *within* the
  interval. The refinement is piecewise: spacing jumps at each boundary.

`d2 < d1` is handled by solving the reversed problem and flipping the result,
so a descending range grades from `d1` rather than toward it.
"""

import numpy as np


def discspace1(d1: float, d2: float, N: int, refine=None,
               kind: str = "linear") -> np.ndarray:
    """``N`` points from ``d1`` to ``d2``, refined where ``refine`` says."""
    if d1 == d2:
        raise ValueError("the endpoints must differ")
    if d2 < d1:
        flipped = None if refine is None else np.asarray(refine)[::-1]
        return discspace1(d2, d1, N, flipped, kind)[::-1]

    if refine is None or len(refine) == 0:
        return np.linspace(d1, d2, N)

    refine = np.atleast_2d(np.asarray(refine, dtype=float))
    if refine.shape[1] != 2:
        raise ValueError(f"refinement must be N by 2, got shape {refine.shape}")
    if np.any(np.diff(refine[:, 0]) <= 0):
        raise ValueError("the first column of the refinement must increase")
    if refine[-1, 0] >= d2:
        raise ValueError("the refinement must end before the final point")
    if refine[0, 0] > d1:
        refine = np.vstack([[d1, 1.0], refine])

    if kind.lower() == "constant":
        span = d2 - d1
        pieces = []
        for i, (start, factor) in enumerate(refine):
            end = d2 if i == len(refine) - 1 else refine[i + 1, 0]
            width = end - start
            count = int(np.ceil(factor * N * width / span))
            step = width / count
            pieces.append(np.linspace(start, end - step, count))
        return np.concatenate(pieces + [np.array([d2])])

    positions = refine[:, 0]
    scaled = np.concatenate([(positions - d1) / (d2 - d1), [1.0]])
    widths = np.diff(np.concatenate([positions, [d2]]))
    density = np.cumsum(np.concatenate([[0.0], refine[:, 1] * widths]))
    density = density / density[-1]
    return (d2 - d1) * np.interp(
        np.linspace(density[0], density[-1], N), density, scaled) + d1


if __name__ == "__main__":

    def test_no_refinement_is_linspace() -> None:
        assert np.allclose(discspace1(0.0, 1.0, 11), np.linspace(0.0, 1.0, 11))
        print("ok test_no_refinement_is_linspace")

    def test_the_endpoints_are_hit_exactly() -> None:
        x = discspace1(0.0, 10.0, 20, [[2.0, 4.0]])
        assert np.isclose(x[0], 0.0) and np.isclose(x[-1], 10.0), (x[0], x[-1])
        print("ok test_the_endpoints_are_hit_exactly")

    def test_the_points_increase() -> None:
        x = discspace1(0.0, 10.0, 25, [[3.0, 5.0]])
        assert np.all(np.diff(x) > 0)
        print("ok test_the_points_increase")

    def test_a_high_factor_packs_points_in() -> None:
        """The whole purpose: more points where the factor is larger."""
        x = discspace1(0.0, 10.0, 41, [[0.0, 1.0], [5.0, 6.0]])
        near = np.count_nonzero((x >= 5.0) & (x <= 7.0))
        far = np.count_nonzero((x >= 1.0) & (x <= 3.0))
        assert near > far, (near, far)
        print(f"ok test_a_high_factor_packs_points_in ({near} vs {far})")

    def test_the_constant_scheme_is_piecewise_uniform() -> None:
        x = discspace1(0.0, 10.0, 40, [[0.0, 1.0], [5.0, 4.0]], kind="constant")
        first = np.diff(x[(x >= 0.5) & (x <= 4.0)])
        second = np.diff(x[(x >= 5.5) & (x <= 9.0)])
        assert np.allclose(first, first[0], rtol=1e-9), "not uniform within"
        assert np.allclose(second, second[0], rtol=1e-9)
        assert second[0] < first[0], (first[0], second[0])
        print("ok test_the_constant_scheme_is_piecewise_uniform")

    def test_the_two_schemes_differ() -> None:
        refine = [[0.0, 1.0], [5.0, 4.0]]
        smooth = discspace1(0.0, 10.0, 30, refine)
        blocky = discspace1(0.0, 10.0, 30, refine, kind="constant")
        assert smooth.size != blocky.size or not np.allclose(smooth, blocky)
        print("ok test_the_two_schemes_differ")

    def test_a_descending_range_is_reversed() -> None:
        x = discspace1(10.0, 0.0, 15, [[2.0, 3.0]])
        assert np.isclose(x[0], 10.0) and np.isclose(x[-1], 0.0)
        assert np.all(np.diff(x) < 0)
        print("ok test_a_descending_range_is_reversed")

    def test_a_non_monotone_refinement_is_rejected() -> None:
        try:
            discspace1(0.0, 10.0, 20, [[5.0, 2.0], [2.0, 3.0]])
        except ValueError as error:
            assert "must increase" in str(error)
            print("ok test_a_non_monotone_refinement_is_rejected")
            return
        raise AssertionError("an out-of-order refinement must be rejected")

    def test_a_refinement_past_the_end_is_rejected() -> None:
        try:
            discspace1(0.0, 10.0, 20, [[12.0, 2.0]])
        except ValueError as error:
            assert "before the final point" in str(error)
            print("ok test_a_refinement_past_the_end_is_rejected")
            return
        raise AssertionError("a refinement beyond d2 must be rejected")

    def test_equal_endpoints_are_rejected() -> None:
        try:
            discspace1(1.0, 1.0, 5)
        except ValueError as error:
            assert "must differ" in str(error)
            print("ok test_equal_endpoints_are_rejected")
            return
        raise AssertionError("a zero-length range must be rejected")

    test_no_refinement_is_linspace()
    test_the_endpoints_are_hit_exactly()
    test_the_points_increase()
    test_a_high_factor_packs_points_in()
    test_the_constant_scheme_is_piecewise_uniform()
    test_the_two_schemes_differ()
    test_a_descending_range_is_reversed()
    test_a_non_monotone_refinement_is_rejected()
    test_a_refinement_past_the_end_is_rejected()
    test_equal_endpoints_are_rejected()
    print("\nAll tests passed!")
