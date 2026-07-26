"""Quadrature and polynomial integration for VEM (MRST `vem/utils`).

The virtual element method builds its local projections from moments - integrals
of monomials over each cell - so it needs exact quadrature on the polygons and
polyhedra of a general grid. This ports the foundation: the polynomial-space
dimension `poly_dim`, the triangle Gauss rules `triangle_quad_rule` (precision 1,
2, 3, 7), and `polygon_int`, which integrates a function over polygonal cells by
triangulating each cell and mapping the reference rule onto every triangle.

These are the building blocks the full `computeVirtualIP` assembles the VEM inner
product from.
"""

from math import comb
from typing import Callable, Sequence, Tuple

import numpy as np
from scipy.spatial import Delaunay


def poly_dim(k: int, dim: int) -> int:
    """Dimension of the space of polynomials of degree <= `k` in `dim`
    variables (MRST `polyDim`): `C(k+dim, k)`, or 0 for `k < 0`."""
    return 0 if k < 0 else comb(k + dim, k)


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


def _cell_node_coords(g, cell: int, dim: int = 2) -> np.ndarray:
    """A cell's corner-node coordinates (gathered from its faces)."""
    face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64).ravel()
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    faces = cell_faces[face_pos[cell]:face_pos[cell + 1]]
    nodes = np.unique(np.concatenate(
        [fnodes[node_pos[f]:node_pos[f + 1]] for f in faces]))
    return np.asarray(g.nodes.coords, dtype=float)[nodes][:, :dim]


def polyhedron_int(g, cells: Sequence[int], f: Callable[[np.ndarray], np.ndarray],
                   k: int) -> np.ndarray:
    """Integrate `f` over polyhedral 3D `cells` with a precision-`k` rule.

    `f` maps `(n, 3)` points to `(n, m)` values; returns `(len(cells), m)`
    (MRST `polyhedronInt`). Each cell is Delaunay-tetrahedralised and the
    reference tet rule is affinely mapped onto every tetrahedron.
    """
    Xq, w, V, vol = tetrahedron_quad_rule(k)
    Vd = V[:-1] - V[1:]                                 # ref edge vectors
    m = np.atleast_2d(np.asarray(f(np.zeros((1, 3))), dtype=float)).shape[1]
    out = np.zeros((len(cells), m))
    for i, c in enumerate(cells):
        X = _cell_node_coords(g, int(c), dim=3)
        tets = Delaunay(X).simplices
        for t in tets:
            Xt = X[t]
            Xd = Xt[:-1] - Xt[1:]                       # physical edge vectors
            A = np.linalg.solve(Vd, Xd)                 # ref -> physical
            b = Xt[0] - V[0] @ A
            det = abs(np.linalg.det(A))
            xhat = Xq @ A + b
            fv = np.atleast_2d(np.asarray(f(xhat), dtype=float))
            out[i] += vol * det * (w @ fv)
    return out


def polygon_int(g, cells: Sequence[int], f: Callable[[np.ndarray], np.ndarray],
                k: int) -> np.ndarray:
    """Integrate `f` over the polygonal 2D `cells` with a precision-`k` rule.

    `f` maps an `(n, 2)` array of points to `(n, m)` values; returns an
    `(len(cells), m)` array of integrals (MRST `polygonInt`).
    """
    Xq, w, _, vol = triangle_quad_rule(k)
    m = np.atleast_2d(np.asarray(f(np.zeros((1, 2))), dtype=float)).shape[1]
    out = np.zeros((len(cells), m))
    for i, c in enumerate(cells):
        X = _cell_node_coords(g, int(c))
        tri = Delaunay(X).simplices
        for t in tri:
            v0, v1, v2 = X[t[0]], X[t[1]], X[t[2]]
            A = np.array([v1 - v0, v2 - v0])           # ref-triangle -> physical
            det = abs(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
            xhat = v0 + Xq @ A
            fv = np.atleast_2d(np.asarray(f(xhat), dtype=float))
            out[i] += vol * det * (w @ fv)
    return out


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_poly_dim():
        assert poly_dim(-1, 2) == 0
        assert poly_dim(0, 2) == 1              # constants
        assert poly_dim(1, 2) == 3              # 1, x, y
        assert poly_dim(2, 2) == 6              # + x^2, xy, y^2
        assert poly_dim(1, 3) == 4
        print("ok test_poly_dim")

    def test_constant_integral_is_cell_area():
        g = compute_geometry(cart_grid([2, 3], [2.0, 3.0]))
        I = polygon_int(g, range(g.cells.num), lambda P: np.ones((P.shape[0], 1)), 1)
        assert np.allclose(I.ravel(), g.cells.volumes), (I.ravel(), g.cells.volumes)
        print("ok test_constant_integral_is_cell_area")

    def test_linear_integral_is_area_times_centroid():
        g = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
        I = polygon_int(g, range(g.cells.num), lambda P: P, 2)   # integrate [x, y]
        expect = g.cells.volumes[:, None] * np.asarray(g.cells.centroids)[:, :2]
        assert np.allclose(I, expect), np.abs(I - expect).max()
        print("ok test_linear_integral_is_area_times_centroid")

    def test_quadratic_is_exact_with_precision_2():
        """A single unit square: integral of x^2 over [0,1]^2 is 1/3, recovered
        exactly by the precision-2 (and higher) rule."""
        g = compute_geometry(cart_grid([1, 1], [1.0, 1.0]))
        I = polygon_int(g, [0], lambda P: (P[:, 0] ** 2)[:, None], 3)
        assert np.isclose(I[0, 0], 1.0 / 3.0), I
        print(f"ok test_quadratic_is_exact_with_precision_2 ({I[0,0]:.6f})")

    def test_high_order_rule_integrates_degree_seven():
        """The precision-7 rule integrates x^6 over the unit square exactly (1/7)."""
        g = compute_geometry(cart_grid([1, 1], [1.0, 1.0]))
        I = polygon_int(g, [0], lambda P: (P[:, 0] ** 6)[:, None], 7)
        assert np.isclose(I[0, 0], 1.0 / 7.0, rtol=1e-10), I
        print(f"ok test_high_order_rule_integrates_degree_seven ({I[0,0]:.6f})")

    def test_polyhedron_constant_integral_is_cell_volume():
        g = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))
        from pymrst.modules.vem.utils.quadrature import polyhedron_int
        I = polyhedron_int(g, range(g.cells.num),
                           lambda P: np.ones((P.shape[0], 1)), 2)
        assert np.allclose(I.ravel(), g.cells.volumes), np.abs(I.ravel() - g.cells.volumes).max()
        print("ok test_polyhedron_constant_integral_is_cell_volume")

    def test_polyhedron_quadratic_is_exact():
        """Integral of x^2 over the unit cube is 1/3, exact with the precision-3
        tetrahedron rule."""
        from pymrst.modules.vem.utils.quadrature import polyhedron_int
        g = compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
        I = polyhedron_int(g, [0], lambda P: (P[:, 0] ** 2)[:, None], 3)
        assert np.isclose(I[0, 0], 1.0 / 3.0, rtol=1e-10), I
        print(f"ok test_polyhedron_quadratic_is_exact ({I[0,0]:.6f})")

    def test_polyhedron_linear_moment():
        g = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
        from pymrst.modules.vem.utils.quadrature import polyhedron_int
        I = polyhedron_int(g, range(g.cells.num), lambda P: P, 2)   # [x, y, z]
        expect = g.cells.volumes[:, None] * np.asarray(g.cells.centroids)
        assert np.allclose(I, expect, atol=1e-12), np.abs(I - expect).max()
        print("ok test_polyhedron_linear_moment")

    test_poly_dim()
    test_constant_integral_is_cell_area()
    test_linear_integral_is_area_times_centroid()
    test_quadratic_is_exact_with_precision_2()
    test_high_order_rule_integrates_degree_seven()
    test_polyhedron_constant_integral_is_cell_volume()
    test_polyhedron_quadratic_is_exact()
    test_polyhedron_linear_moment()
    print("\nAll tests passed!")
