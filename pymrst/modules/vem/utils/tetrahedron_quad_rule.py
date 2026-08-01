"""Tetrahedron Gauss rules (MRST ``vem/utils/tetrahedronQuadRule.m``).

Quadrature on MRST's reference tetrahedron, used by `polyhedron_int` after
each cell has been tetrahedralised. Precisions 2 and 3 are provided, matching
MRST.

The reference element is **equilateral and centred on the origin**, not the
unit corner tetrahedron the triangle rules use. That is MRST's choice and it
matters here: the quadrature points are given in those coordinates, so pairing
them with a unit-corner reference - the obvious assumption - silently
integrates over the wrong domain. `polyhedron_int` maps from these vertices,
which is why the rule returns `V` alongside the points.
"""

from typing import Tuple

import numpy as np

_S3, _S6 = np.sqrt(3.0), np.sqrt(6.0)
_TET_V = np.array([[-1.0, -1.0 / _S3, -1.0 / _S6],
                   [0.0, 2.0 / _S3, -1.0 / _S6],
                   [1.0, -1.0 / _S3, -1.0 / _S6],
                   [0.0, 0.0, 3.0 / _S6]])
_TET_VOL = np.sqrt(np.sqrt(8.0) / 3.0)

_TET_RULES = {
    2: (np.array([
        [-.2677028585910073, .5939017006199488e-01, -.3969426941142150],
        [.1510931841533142, .4354087732476309, .2151719254306919],
        [-.1367699195237390, -.3280341115590410, .2950846519937133],
        [.8067449309051964, -.3400977314285288, -.3430378951002550]]),
        np.array([.2918008865477151, .2706884392356724,
                  .3098349601753470, .9865925745591227e-01])),
    3: (np.array([
        [-.1685037180276000, .1910914916271708, -.3896267314585163],
        [.2783799427534418e-01, -.2304932838839657e-01, 0.5481350663241830],
        [-.3512216177343445, .1835144026339993, .5147815330343534e-01],
        [.4308532463549043, -.2474715823180446, -.1683315641007033],
        [-.4676763747967377, -.4235250827264375, -.1586973077889307],
        [.1492831253848431, .6397847685164516, -.1080219253055393]]),
        np.array([.1287213727402025, .2179034339695993, .1243503792062836,
                  .2446917182410072, .1365439875826512, .1187726516749031])),
}


def tetrahedron_quad_rule(k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return `(Xq, w, V, vol)` for a tetrahedron rule of precision `k` on MRST's
    equilateral reference tet (`tetrahedronQuadRule`; supports k = 2, 3)."""
    if k not in _TET_RULES:
        raise ValueError("tetrahedron rules of precision 2 and 3 only")
    Xq, w = _TET_RULES[k]
    return Xq.copy(), w.copy(), _TET_V.copy(), _TET_VOL


tetrahedronQuadRule = tetrahedron_quad_rule


if __name__ == "__main__":

    def test_the_weights_and_volume_together_give_the_reference_volume() -> None:
        """`vol * sum(w)` is the volume - `sum(w)` alone is not 1.

        MRST splits the normalisation between the two returned values, and
        `vol` is the square root of the geometric volume rather than the volume
        itself. A caller that assumes unit weights, or that `vol` is the
        volume, is wrong by a factor that no constant-function test catches
        because both errors cancel there.
        """
        for k in (2, 3):
            _, w, V, vol = tetrahedron_quad_rule(k)
            geometric = abs(np.linalg.det(V[1:] - V[0])) / 6.0
            assert np.isclose(vol * w.sum(), geometric, rtol=1e-12), (k, vol)
            assert not np.isclose(w.sum(), 1.0), "weights are not normalised"
            assert not np.isclose(vol, geometric), "vol is not the volume"
        print("ok test_the_weights_and_volume_together_give_the_reference_volume")

    def test_a_linear_function_integrates_to_volume_times_centroid_value() -> None:
        """Exact and analytic: no second quadrature rule involved."""
        for k in (2, 3):
            Xq, w, V, vol = tetrahedron_quad_rule(k)
            geometric = abs(np.linalg.det(V[1:] - V[0])) / 6.0
            centroid = V.mean(axis=0)
            for axis in range(3):
                def f(X, a=axis):
                    return 2.0 + 3.0 * X[:, a]
                got = vol * float(w @ f(Xq))
                want = geometric * (2.0 + 3.0 * centroid[axis])
                assert np.isclose(got, want, rtol=1e-12), (k, axis, got, want)
        print("ok test_a_linear_function_integrates_to_volume_times_centroid_value")

    def test_the_two_rules_agree_on_a_quadratic() -> None:
        """Both claim degree-2 exactness, so they must agree to machine precision.

        They share no points and no weights, so agreement is evidence about
        the tables rather than about a formula they have in common.
        """
        def f(X):
            return 1.0 + X[:, 0] - 2.0 * X[:, 1] + X[:, 0] * X[:, 2]

        values = []
        for k in (2, 3):
            Xq, w, _, vol = tetrahedron_quad_rule(k)
            values.append(vol * float(w @ f(Xq)))
        assert np.isclose(values[0], values[1], rtol=1e-10), values
        print("ok test_the_two_rules_agree_on_a_quadratic")

    def test_the_reference_tetrahedron_is_equilateral() -> None:
        """Every edge the same length - not the unit corner tetrahedron."""
        _, _, V, _ = tetrahedron_quad_rule(2)
        lengths = [np.linalg.norm(V[i] - V[j])
                   for i in range(4) for j in range(i + 1, 4)]
        assert np.allclose(lengths, lengths[0], rtol=1e-12), lengths
        print("ok test_the_reference_tetrahedron_is_equilateral")

    def test_the_reference_element_is_centred_on_the_origin() -> None:
        _, _, V, _ = tetrahedron_quad_rule(3)
        assert np.allclose(V.mean(axis=0), 0.0, atol=1e-12), V.mean(axis=0)
        print("ok test_the_reference_element_is_centred_on_the_origin")

    def test_the_returned_arrays_are_copies() -> None:
        _, w, _, _ = tetrahedron_quad_rule(2)
        original = w.sum()
        w *= 2.0
        _, again, _, _ = tetrahedron_quad_rule(2)
        assert np.isclose(again.sum(), original), again.sum()
        print("ok test_the_returned_arrays_are_copies")

    def test_an_unsupported_precision_is_rejected() -> None:
        try:
            tetrahedron_quad_rule(1)
        except ValueError as error:
            assert "2 and 3" in str(error)
            print("ok test_an_unsupported_precision_is_rejected")
            return
        raise AssertionError("only rules 2 and 3 exist")

    test_the_weights_and_volume_together_give_the_reference_volume()
    test_a_linear_function_integrates_to_volume_times_centroid_value()
    test_the_two_rules_agree_on_a_quadratic()
    test_the_reference_tetrahedron_is_equilateral()
    test_the_reference_element_is_centred_on_the_origin()
    test_the_returned_arrays_are_copies()
    test_an_unsupported_precision_is_rejected()
    print("\nAll tests passed!")
