"""Triangle Gauss rules (MRST ``vem/utils/triangleQuadRule.m``).

Quadrature on the reference triangle with vertices (0,0), (1,0), (0,1), used
by `polygon_int` and `polygon_int_3d` after each polygon has been
triangulated.

Precisions 1, 2, 3 and 7 are provided, matching MRST. The precision-3 rule
carries a **negative** weight at the centroid - that is correct, not a
transcription error: it is the classical four-point Hammer-Stroud rule, and
the negative centroid weight is what buys degree-3 exactness from four points.
A rule with all-positive weights would need more.
"""

from typing import Tuple

import numpy as np

_TRI_V = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
_TRI_VOL = 0.5

_TRI_RULES = {
    1: (np.array([[1 / 3, 1 / 3]]), np.array([1.0])),
    2: (np.array([[0.5, 0.0], [0.5, 0.5], [0.0, 0.5]]),
        np.array([1 / 3, 1 / 3, 1 / 3])),
    3: (np.array([[1 / 3, 1 / 3], [0.6, 0.2], [0.2, 0.6], [0.2, 0.2]]),
        np.array([-0.5625, 0.52083333333333333, 0.52083333333333333,
                  0.52083333333333333])),
    7: (np.array([
        [0.333333333333333, 0.333333333333333],
        [0.479308067841923, 0.260345966079038],
        [0.260345966079038, 0.479308067841923],
        [0.260345966079038, 0.260345966079038],
        [0.869739794195568, 0.065130102902216],
        [0.065130102902216, 0.869739794195568],
        [0.065130102902216, 0.065130102902216],
        [0.638444188569809, 0.312865496004875],
        [0.638444188569809, 0.048690315425316],
        [0.312865496004875, 0.638444188569809],
        [0.312865496004875, 0.048690315425316],
        [0.048690315425316, 0.638444188569809],
        [0.048690315425316, 0.312865496004875]]),
        np.array([-0.149570044467670,
                  0.175615257433204, 0.175615257433204, 0.175615257433204,
                  0.053347235608839, 0.053347235608839, 0.053347235608839,
                  0.077113760890257, 0.077113760890257, 0.077113760890257,
                  0.077113760890257, 0.077113760890257, 0.077113760890257])),
}


def triangle_quad_rule(k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return `(Xq, w, V, vol)` for a triangle Gauss rule of precision `k`
    (MRST `triangleQuadRule`; supports k = 1, 2, 3, 7)."""
    if k not in _TRI_RULES:
        raise ValueError("triangle rules of precision 1, 2, 3 and 7 only")
    Xq, w = _TRI_RULES[k]
    return Xq.copy(), w.copy(), _TRI_V.copy(), _TRI_VOL


triangleQuadRule = triangle_quad_rule


if __name__ == "__main__":

    def _integrate(k: int, f) -> float:
        """Integrate f over the reference triangle with the precision-k rule."""
        Xq, w, _, vol = triangle_quad_rule(k)
        return vol * float(w @ f(Xq))

    def test_every_rule_integrates_a_constant() -> None:
        """The weights sum to 1, so a constant gives the reference area, 1/2."""
        for k in (1, 2, 3, 7):
            assert np.isclose(_integrate(k, lambda X: np.ones(len(X))), 0.5), k
        print("ok test_every_rule_integrates_a_constant")

    def test_each_rule_is_exact_to_its_stated_precision() -> None:
        """Integral of x^p over the reference triangle is 1/((p+1)(p+2))."""
        for k in (1, 2, 3, 7):
            for p in range(k + 1):
                got = _integrate(k, lambda X, p=p: X[:, 0] ** p)
                want = 1.0 / ((p + 1) * (p + 2))
                assert np.isclose(got, want, rtol=1e-12), (k, p, got, want)
        print("ok test_each_rule_is_exact_to_its_stated_precision")

    def test_the_precision_three_rule_has_a_negative_weight() -> None:
        """Deliberate: it is what makes four points reach degree three."""
        _, w, _, _ = triangle_quad_rule(3)
        assert np.any(w < 0), w
        assert np.isclose(w.sum(), 1.0), w.sum()
        print("ok test_the_precision_three_rule_has_a_negative_weight")

    def test_points_lie_inside_the_reference_triangle() -> None:
        for k in (1, 2, 3, 7):
            Xq, _, _, _ = triangle_quad_rule(k)
            assert np.all(Xq >= -1e-12), k
            assert np.all(Xq.sum(axis=1) <= 1 + 1e-12), k
        print("ok test_points_lie_inside_the_reference_triangle")

    def test_the_returned_arrays_are_copies() -> None:
        """A caller scaling the weights must not corrupt the module tables."""
        _, w, _, _ = triangle_quad_rule(1)
        w *= 2.0
        _, again, _, _ = triangle_quad_rule(1)
        assert np.isclose(again.sum(), 1.0), again
        print("ok test_the_returned_arrays_are_copies")

    def test_an_unsupported_precision_is_rejected() -> None:
        try:
            triangle_quad_rule(4)
        except ValueError as error:
            assert "1, 2, 3 and 7" in str(error)
            print("ok test_an_unsupported_precision_is_rejected")
            return
        raise AssertionError("only rules 1, 2, 3 and 7 exist")

    test_every_rule_integrates_a_constant()
    test_each_rule_is_exact_to_its_stated_precision()
    test_the_precision_three_rule_has_a_negative_weight()
    test_points_lie_inside_the_reference_triangle()
    test_the_returned_arrays_are_copies()
    test_an_unsupported_precision_is_rejected()
    print("\nAll tests passed!")
