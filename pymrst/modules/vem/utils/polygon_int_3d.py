"""Integrate over planar faces in 3-D (MRST ``vem/utils/polygonInt3D.m``).

A face of a 3-D grid is a polygon embedded in space, so it cannot be
triangulated in the global frame directly. Each face is mapped into its own
in-plane coordinate system - `v1` along the first edge, `v2 = normal x v1` -
triangulated there, and the reference triangle rule is mapped onto every
triangle before the points are lifted back to 3-D for evaluation.

**The integral is taken over the face flattened into that plane**, which
matters for corner-point grids where a quadrilateral face is often not exactly
planar. MRST does the same, so a warped face contributes its projected area
rather than its true surface area, and the two codes agree because they make
the same approximation - not because the face was planar.
"""

from typing import Callable, Sequence

import numpy as np
from scipy.spatial import Delaunay

from pymrst.modules.vem.utils.triangle_quad_rule import triangle_quad_rule

def polygon_int_3d(g, faces: Sequence[int], f: Callable[[np.ndarray], np.ndarray],
                   k: int) -> np.ndarray:
    """Integrate `f` over 3-D planar `faces` with a precision-`k` rule.

    `f` maps `(n, 3)` global points to `(n, m)` values; returns `(len(faces), m)`
    (MRST `polygonInt3D`). Each face is mapped to its own 2-D coordinate system
    (v1 along the first edge, v2 = normal x v1), triangulated there, and the
    reference triangle rule is affinely mapped onto every triangle - so, as in
    MRST, the integral is taken over the face flattened into that plane.
    """
    Xq, w, _, vol = triangle_quad_rule(k)
    node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    fnodes = np.asarray(g.faces.nodes, dtype=np.int64)
    coords = np.asarray(g.nodes.coords, dtype=float)
    normals = np.asarray(g.faces.normals, dtype=float)
    m = np.atleast_2d(np.asarray(f(np.zeros((1, 3))), dtype=float)).shape[1]
    out = np.zeros((len(faces), m))
    for i, face in enumerate(faces):
        nds = fnodes[node_pos[face]:node_pos[face + 1]]
        X = coords[nds]
        v1 = X[1] - X[0]
        v1 /= np.linalg.norm(v1)
        v2 = np.cross(normals[face], v1)
        v2 /= np.linalg.norm(v2)
        T = np.column_stack([v1, v2])                  # 3x2 global->local
        b0 = X[0]
        X2 = (X - b0) @ T                              # face nodes in local 2-D
        for t in Delaunay(X2).simplices:
            v0, e1, e2 = X2[t[0]], X2[t[1]] - X2[t[0]], X2[t[2]] - X2[t[0]]
            A2 = np.array([e1, e2])
            det = abs(A2[0, 0] * A2[1, 1] - A2[0, 1] * A2[1, 0])
            xhat2 = v0 + Xq @ A2                        # local quad points
            xhat = xhat2 @ T.T + b0                     # back to global 3-D
            fv = np.atleast_2d(np.asarray(f(xhat), dtype=float))
            out[i] += vol * det * (w @ fv)
    return out


polygonInt3D = polygon_int_3d


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry

    def test_a_constant_integrates_to_the_face_area() -> None:
        g = compute_geometry(cart_grid([1, 1, 1], [2.0, 3.0, 4.0]))
        got = polygon_int_3d(g, range(g.faces.num),
                             lambda P: np.ones((P.shape[0], 1)), 1).ravel()
        assert np.allclose(got, g.faces.areas), np.abs(got - g.faces.areas).max()
        print("ok test_a_constant_integrates_to_the_face_area")

    def test_a_quadratic_over_a_face_is_exact() -> None:
        """x^2 over the x = 2 face of a 2x3x4 box is 4 * 12 = 48."""
        g = compute_geometry(cart_grid([1, 1, 1], [2.0, 3.0, 4.0]))
        centroids = np.asarray(g.faces.centroids)
        face = int(np.flatnonzero(np.abs(centroids[:, 0] - 2.0) < 1e-9)[0])
        got = polygon_int_3d(g, [face], lambda P: (P[:, 0] ** 2)[:, None], 3)
        assert np.isclose(got[0, 0], 48.0, rtol=1e-10), got
        print("ok test_a_quadratic_over_a_face_is_exact")

    def test_a_linear_function_gives_area_times_centroid() -> None:
        g = compute_geometry(cart_grid([1, 1, 1], [2.0, 3.0, 4.0]))
        got = polygon_int_3d(g, range(g.faces.num), lambda P: P, 2)
        want = np.asarray(g.faces.areas)[:, None] * np.asarray(g.faces.centroids)
        assert np.allclose(got, want, atol=1e-10), np.abs(got - want).max()
        print("ok test_a_linear_function_gives_area_times_centroid")

    def test_the_result_is_independent_of_the_face_orientation() -> None:
        """Every face of a cube gets its own in-plane frame; all give the area."""
        g = compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
        got = polygon_int_3d(g, range(g.faces.num),
                             lambda P: np.ones((len(P), 1)), 1).ravel()
        assert np.allclose(got, 1.0), got
        print("ok test_the_result_is_independent_of_the_face_orientation")

    def test_several_components_are_integrated_at_once() -> None:
        g = compute_geometry(cart_grid([1, 1, 1], [1.0, 1.0, 1.0]))
        got = polygon_int_3d(g, range(g.faces.num),
                             lambda P: np.column_stack([np.ones(len(P)), P]), 2)
        assert got.shape == (g.faces.num, 4), got.shape
        print("ok test_several_components_are_integrated_at_once")

    def test_interior_faces_are_integrated_too() -> None:
        g = compute_geometry(cart_grid([2, 1, 1], [2.0, 1.0, 1.0]))
        centroids = np.asarray(g.faces.centroids)
        interior = int(np.flatnonzero(np.abs(centroids[:, 0] - 1.0) < 1e-9)[0])
        got = polygon_int_3d(g, [interior],
                             lambda P: np.ones((len(P), 1)), 1)[0, 0]
        assert np.isclose(got, 1.0), got
        print("ok test_interior_faces_are_integrated_too")

    test_a_constant_integrates_to_the_face_area()
    test_a_quadratic_over_a_face_is_exact()
    test_a_linear_function_gives_area_times_centroid()
    test_the_result_is_independent_of_the_face_orientation()
    test_several_components_are_integrated_at_once()
    test_interior_faces_are_integrated_too()
    print("\nAll tests passed!")
