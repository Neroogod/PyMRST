"""Integrate over polygonal cells (MRST ``vem/utils/polygonInt.m``).

Each 2-D cell is triangulated and the reference triangle rule is mapped
affinely onto every triangle, so the result is exact for polynomials up to the
rule's precision on any convex polygon.

The triangulation is Delaunay over the cell's corner nodes, which is exact for
a convex cell and *wrong for a non-convex one* - Delaunay fills the hull, so a
re-entrant corner is integrated as though it were solid. MRST has the same
limitation. Grids from `cartGrid`, `pebi` and corner-point processing are
convex cell by cell, which is why it holds in practice.
"""

from typing import Callable, Sequence

import numpy as np
from scipy.spatial import Delaunay

from pymrst.modules.vem.utils._cell_geometry import cell_node_coords
from pymrst.modules.vem.utils.triangle_quad_rule import triangle_quad_rule

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
        X = cell_node_coords(g, int(c))
        tri = Delaunay(X).simplices
        for t in tri:
            v0, v1, v2 = X[t[0]], X[t[1]], X[t[2]]
            A = np.array([v1 - v0, v2 - v0])           # ref-triangle -> physical
            det = abs(A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0])
            xhat = v0 + Xq @ A
            fv = np.atleast_2d(np.asarray(f(xhat), dtype=float))
            out[i] += vol * det * (w @ fv)
    return out


polygonInt = polygon_int


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_a_constant_integrates_to_the_cell_area() -> None:
        g = compute_geometry(cart_grid([2, 3], [2.0, 3.0]))
        got = polygon_int(g, range(g.cells.num),
                          lambda P: np.ones((P.shape[0], 1)), 1)
        assert np.allclose(got.ravel(), g.cells.volumes), got.ravel()
        print("ok test_a_constant_integrates_to_the_cell_area")

    def test_a_linear_function_gives_area_times_centroid() -> None:
        g = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
        got = polygon_int(g, range(g.cells.num), lambda P: P, 2)
        want = g.cells.volumes[:, None] * np.asarray(g.cells.centroids)[:, :2]
        assert np.allclose(got, want), np.abs(got - want).max()
        print("ok test_a_linear_function_gives_area_times_centroid")

    def test_a_quadratic_is_exact_with_the_precision_three_rule() -> None:
        g = compute_geometry(cart_grid([1, 1], [1.0, 1.0]))
        got = polygon_int(g, [0], lambda P: (P[:, 0] ** 2)[:, None], 3)
        assert np.isclose(got[0, 0], 1.0 / 3.0), got
        print("ok test_a_quadratic_is_exact_with_the_precision_three_rule")

    def test_degree_seven_is_exact_with_the_precision_seven_rule() -> None:
        g = compute_geometry(cart_grid([1, 1], [1.0, 1.0]))
        got = polygon_int(g, [0], lambda P: (P[:, 0] ** 6)[:, None], 7)
        assert np.isclose(got[0, 0], 1.0 / 7.0, rtol=1e-10), got
        print("ok test_degree_seven_is_exact_with_the_precision_seven_rule")

    def test_a_low_order_rule_is_wrong_on_a_high_degree_polynomial() -> None:
        """The precision argument is doing something, not decoration."""
        g = compute_geometry(cart_grid([1, 1], [1.0, 1.0]))
        coarse = polygon_int(g, [0], lambda P: (P[:, 0] ** 6)[:, None], 1)
        assert not np.isclose(coarse[0, 0], 1.0 / 7.0, rtol=1e-3), coarse
        print("ok test_a_low_order_rule_is_wrong_on_a_high_degree_polynomial")

    def test_several_components_are_integrated_at_once() -> None:
        g = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        got = polygon_int(g, range(g.cells.num),
                          lambda P: np.column_stack([np.ones(len(P)), P]), 2)
        assert got.shape == (4, 3), got.shape
        assert np.allclose(got[:, 0], g.cells.volumes)
        print("ok test_several_components_are_integrated_at_once")

    def test_the_integral_is_additive_over_cells() -> None:
        """Two cells must sum to the same as the single cell covering them."""
        fine = compute_geometry(cart_grid([2, 1], [2.0, 1.0]))
        whole = compute_geometry(cart_grid([1, 1], [2.0, 1.0]))
        f = lambda P: (P[:, 0] ** 2)[:, None]
        assert np.isclose(polygon_int(fine, range(2), f, 3).sum(),
                          polygon_int(whole, [0], f, 3).sum())
        print("ok test_the_integral_is_additive_over_cells")

    test_a_constant_integrates_to_the_cell_area()
    test_a_linear_function_gives_area_times_centroid()
    test_a_quadratic_is_exact_with_the_precision_three_rule()
    test_degree_seven_is_exact_with_the_precision_seven_rule()
    test_a_low_order_rule_is_wrong_on_a_high_degree_polynomial()
    test_several_components_are_integrated_at_once()
    test_the_integral_is_additive_over_cells()
    print("\nAll tests passed!")
