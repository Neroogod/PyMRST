"""Integrate over polyhedral cells (MRST ``vem/utils/polyhedronInt.m``).

Each 3-D cell is Delaunay-tetrahedralised and the reference tetrahedron rule
is mapped affinely onto every tetrahedron.

The mapping is built from *edge vectors* rather than from vertices directly,
because MRST's reference tetrahedron is equilateral and centred on the origin
rather than sitting in a corner - see `tetrahedron_quad_rule`. Solving
`Vd A = Xd` for the linear part and recovering the offset from one vertex is
what makes the rule's own coordinates irrelevant to the caller.

Same convexity caveat as `polygon_int`: Delaunay fills the convex hull, so a
non-convex cell is integrated as though it were solid.
"""

from typing import Callable, Sequence

import numpy as np
from scipy.spatial import Delaunay

from pymrst.modules.vem.utils._cell_geometry import cell_node_coords
from pymrst.modules.vem.utils.tetrahedron_quad_rule import tetrahedron_quad_rule

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
        X = cell_node_coords(g, int(c), dim=3)
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


polyhedronInt = polyhedron_int


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_a_constant_integrates_to_the_cell_volume() -> None:
        g = compute_geometry(cart_grid([2, 2, 2], [2.0, 2.0, 2.0]))
        got = polyhedron_int(g, range(g.cells.num),
                             lambda P: np.ones((P.shape[0], 1)), 2)
        assert np.allclose(got.ravel(), g.cells.volumes), \
            np.abs(got.ravel() - g.cells.volumes).max()
        print("ok test_a_constant_integrates_to_the_cell_volume")

    def test_a_linear_function_gives_volume_times_centroid() -> None:
        g = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
        got = polyhedron_int(g, range(g.cells.num), lambda P: P, 2)
        want = g.cells.volumes[:, None] * np.asarray(g.cells.centroids)
        assert np.allclose(got, want, atol=1e-12), np.abs(got - want).max()
        print("ok test_a_linear_function_gives_volume_times_centroid")

    def test_a_quadratic_is_exact_with_the_precision_three_rule() -> None:
        g = compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
        got = polyhedron_int(g, [0], lambda P: (P[:, 0] ** 2)[:, None], 3)
        assert np.isclose(got[0, 0], 1.0 / 3.0, rtol=1e-10), got
        print("ok test_a_quadratic_is_exact_with_the_precision_three_rule")

    def test_a_non_cubic_box_is_handled() -> None:
        """The affine map must carry anisotropic scaling, not just translation."""
        g = compute_geometry(cart_grid([1, 1, 1], [2.0, 3.0, 4.0]))
        got = polyhedron_int(g, [0], lambda P: np.ones((len(P), 1)), 2)
        assert np.isclose(got[0, 0], 24.0), got
        print("ok test_a_non_cubic_box_is_handled")

    def test_several_components_are_integrated_at_once() -> None:
        g = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
        got = polyhedron_int(g, range(g.cells.num),
                             lambda P: np.column_stack([np.ones(len(P)), P]), 2)
        assert got.shape == (2, 4), got.shape
        assert np.allclose(got[:, 0], g.cells.volumes)
        print("ok test_several_components_are_integrated_at_once")

    def test_the_integral_is_additive_over_cells() -> None:
        fine = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
        whole = compute_geometry(cart_grid([1, 1, 1], [2.0, 1.0, 1.0]))
        f = lambda P: (P[:, 0] ** 2)[:, None]
        assert np.isclose(polyhedron_int(fine, range(2), f, 3).sum(),
                          polyhedron_int(whole, [0], f, 3).sum(), rtol=1e-10)
        print("ok test_the_integral_is_additive_over_cells")

    test_a_constant_integrates_to_the_cell_volume()
    test_a_linear_function_gives_volume_times_centroid()
    test_a_quadratic_is_exact_with_the_precision_three_rule()
    test_a_non_cubic_box_is_handled()
    test_several_components_are_integrated_at_once()
    test_the_integral_is_additive_over_cells()
    print("\nAll tests passed!")
