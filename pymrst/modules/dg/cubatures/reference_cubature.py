"""Reference-element cubature rules for the DG module (MRST get*CubaturePoints).

Returns exact points and weights for a cubature of precision `k` on the reference
line, square, triangle, cube or tetrahedron, looked up from MRST's tables. A rule
of precision `k` integrates polynomials of total degree <= `k` exactly. Weights
are normalised to sum to 1 (the reference-element average), matching MRST.

MRST bug preserved for faithfulness: the cube rule for k in {2,3} lists its
second point as (1,0,0) instead of (-1,0,0), so it does NOT integrate monomials
odd in x exactly. PyMRST reproduces MRST's table verbatim (so the two agree);
callers needing a correct low-order cube rule should request k >= 4.
"""

from typing import Tuple

import numpy as np

from pymrst.modules.dg.cubatures import _cubature_tables as _T


def _lookup(rules, xR, k: int) -> Tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    for max_k, x, w in rules:
        if k <= max_k:
            return x.copy(), w.copy(), x.shape[0], xR
    raise ValueError(f"no cubature rule of precision {k} available "
                     f"(max {rules[-1][0]})")


def get_line_cubature(k: int):
    """Precision-`k` cubature on the reference line [-1, 1] (Gauss-Legendre)."""
    return _lookup(_T._LINE_RULES, _T._LINE_XR, k)


def get_square_cubature(k: int):
    """Precision-`k` cubature on the reference square [-1, 1]^2."""
    return _lookup(_T._SQUARE_RULES, _T._SQUARE_XR, k)


def get_triangle_cubature(k: int):
    """Precision-`k` cubature on the reference triangle (-1,-1),(1,-1),(-1,1)."""
    return _lookup(_T._TRIANGLE_RULES, _T._TRIANGLE_XR, k)


def get_cube_cubature(k: int):
    """Precision-`k` cubature on the reference cube [-1, 1]^3."""
    return _lookup(_T._CUBE_RULES, _T._CUBE_XR, k)


def get_tetrahedron_cubature(k: int):
    """Precision-`k` cubature on the reference tetrahedron."""
    return _lookup(_T._TET_RULES, _T._TET_XR, k)


if __name__ == "__main__":
    from itertools import product

    # MRST normalises the returned weights to sum to 1 (the reference-element
    # average), so `w @ f` is the mean of f over the element: the exact integral
    # is `measure * (w @ f)`.
    def test_weights_sum_to_one():
        for getter in (get_line_cubature, get_square_cubature,
                       get_triangle_cubature, get_cube_cubature,
                       get_tetrahedron_cubature):
            _, w, _, _ = getter(3)
            assert abs(w.sum() - 1.0) < 1e-12, (getter.__name__, w.sum())
        print("ok test_weights_sum_to_one")

    def test_line_matches_gauss_legendre():
        # MRST's line rule is Gauss-Legendre on [-1, 1] with weights halved.
        from numpy.polynomial.legendre import leggauss
        for k in (1, 3, 5, 7, 9):
            x, w, n, _ = get_line_cubature(k)
            gx, gw = leggauss(n)
            assert np.allclose(np.sort(x.ravel()), np.sort(gx))
            assert np.allclose(np.sort(w), np.sort(gw / 2))
        print("ok test_line_matches_gauss_legendre")

    def test_reference_rules_integrate_monomials_exactly():
        # measure * (w @ monomial) equals the analytic integral over the element,
        # for every monomial up to the rule's precision. (Cube k in {2,3} skipped:
        # documented MRST bug - see test_cube_low_order_mrst_bug.)
        cases = [(get_square_cubature, 2, 4.0, (2, 4, 5)),
                 (get_cube_cubature, 3, 8.0, (4, 5))]
        for getter, dim, measure, ks in cases:
            for k in ks:
                x, w = getter(k)[:2]
                for exps in product(range(k + 1), repeat=dim):
                    if sum(exps) > k:
                        continue
                    num = measure * float(w @ np.prod(x ** np.array(exps), axis=1))
                    exact = 1.0
                    for e in exps:
                        exact *= (1 - (-1) ** (e + 1)) / (e + 1)
                    assert abs(num - exact) < 1e-10, (getter.__name__, k, exps)
        print("ok test_reference_rules_integrate_monomials_exactly")

    def test_cube_low_order_mrst_bug():
        # MRST's cube k<=3 rule has (1,0,0) twice (should be +/-1), so it fails
        # for x but is fine for even powers. PyMRST preserves this verbatim.
        x, w, _, _ = get_cube_cubature(3)
        assert np.count_nonzero((x == [1, 0, 0]).all(axis=1)) == 2   # the bug
        assert abs(8.0 * (w @ x[:, 0])) > 0.1                        # int x wrong
        assert abs(8.0 * (w @ x[:, 0] ** 2) - 8.0 / 3.0) < 1e-10     # int x^2 ok
        print("ok test_cube_low_order_mrst_bug (documented)")

    def test_triangle_and_tet_centroid():
        # Weights sum to 1, so w @ x is the centroid: (-1/3,-1/3) and (-1/2,...).
        x, w, _, _ = get_triangle_cubature(2)
        assert np.allclose(w @ x, [-1 / 3, -1 / 3], atol=1e-12)
        xt, wt, _, _ = get_tetrahedron_cubature(2)
        assert np.allclose(wt @ xt, [-0.5, -0.5, -0.5], atol=1e-12), wt @ xt
        print("ok test_triangle_and_tet_centroid")

    test_weights_sum_to_one()
    test_line_matches_gauss_legendre()
    test_reference_rules_integrate_monomials_exactly()
    test_cube_low_order_mrst_bug()
    test_triangle_and_tet_centroid()
    print("\nAll tests passed!")
