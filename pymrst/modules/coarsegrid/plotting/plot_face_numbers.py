"""Label faces with their numbers (MRST ``plotFaceNumbers.m``).

The face counterpart of `plot_cell_numbers`. On a coarse grid it labels coarse
faces at the area-weighted centroid of the fine faces behind them, which is how
MRST's examples show which fine faces a coarse connection stands for.

Positions come from `face_number_labels` so they can be checked without a
display, and the same two departures from MATLAB apply as in
`plot_cell_numbers`: labels are 0-based unless `base=1` is passed, and coarse
geometry comes from `coarsen_geometry` rather than from MRST's private helper.

That helper is worth naming precisely, because both numbering plots carry a copy
of it and both are wrong in the same way. It computes a coarse face centroid as

    sum(fine centroids) / sum(fine areas)

where the numerator has no area weighting, so the result is only the centroid
when every fine face has unit area. On a unit Cartesian grid it happens to be
right, which is why the examples look correct. `coarsenGeometry` weights the
numerator properly, and this port calls that.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pymrst.modules.coarsegrid.plotting.plot_cell_numbers import _to_three


def face_number_labels(g, faces: Sequence[int] | None = None, base: int = 0):
    """Return ``(positions, labels)`` for the given faces or coarse faces."""
    centroids = _face_centroids(g)
    n = centroids.shape[0]
    if faces is None:
        faces = np.arange(n, dtype=np.int64)
    faces = np.asarray(faces, dtype=np.int64).ravel()
    if faces.size and (faces.min() < 0 or faces.max() >= n):
        raise ValueError(f"face numbers must lie in 0 to {n - 1}")
    return centroids[faces], [str(int(f) + base) for f in faces]


def _face_centroids(g) -> np.ndarray:
    if hasattr(g, "partition") and hasattr(g, "n_blocks"):
        if getattr(g, "face_centroids", None) is None:
            from pymrst.modules.coarsegrid.utils.coarsen_geometry import (
                coarsen_geometry,
            )
            coarsen_geometry(g)
        if g.face_centroids is None:
            raise ValueError("this coarse grid carries no face connectivity")
        return np.asarray(g.face_centroids, dtype=float)
    if getattr(g.faces, "centroids", None) is None:
        from pymrst.gridprocessing import compute_geometry
        g = compute_geometry(g)
    return np.asarray(g.faces.centroids, dtype=float)


def plot_face_numbers(g, faces: Sequence[int] | None = None, base: int = 0,
                      plotter=None, **kwargs) -> Any:
    """Draw the labels, returning the plotter, or ``None`` without a backend."""
    positions, labels = face_number_labels(g, faces, base)
    try:
        import pyvista as pv
    except ImportError:
        return None
    plotter = plotter or pv.Plotter()
    plotter.add_point_labels(_to_three(positions), labels, **kwargs)
    return plotter


__all__ = ["face_number_labels", "plot_face_numbers"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
    from pymrst.modules.coarsegrid.partition_cart_grid import partition_cart_grid

    def _fine(dims=(3, 2, 1)):
        return compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))

    def _coarse(dims=(4, 4, 1), part=(2, 2, 1)):
        g = _fine(dims)
        p = partition_cart_grid(list(dims), list(part))
        return g, p, generate_coarse_grid(g, p)

    def test_every_face_gets_a_label_at_its_own_centroid():
        g = _fine()
        pos, labels = face_number_labels(g)
        assert len(labels) == g.faces.num
        assert np.allclose(pos, np.asarray(g.faces.centroids))
        assert labels[0] == "0" and labels[-1] == str(g.faces.num - 1)
        print(f"ok test_every_face_gets_a_label_at_its_own_centroid "
              f"({g.faces.num} faces)")

    def test_a_subset_keeps_its_own_numbers():
        g = _fine()
        pos, labels = face_number_labels(g, [7, 2])
        assert labels == ["7", "2"], labels
        assert np.allclose(pos[0], np.asarray(g.faces.centroids)[7])
        print("ok test_a_subset_keeps_its_own_numbers")

    def test_labels_are_zero_based_by_default():
        g = _fine()
        assert face_number_labels(g)[1][0] == "0"
        assert face_number_labels(g, base=1)[1][0] == "1"
        print("ok test_labels_are_zero_based_by_default")

    def test_a_coarse_grid_labels_coarse_faces():
        g, _, cg = _coarse()
        pos, labels = face_number_labels(cg)
        assert len(labels) == cg.num_faces, (len(labels), cg.num_faces)
        assert cg.num_faces < g.faces.num
        print(f"ok test_a_coarse_grid_labels_coarse_faces "
              f"({cg.num_faces} coarse of {g.faces.num} fine)")

    def test_a_coarse_label_sits_among_its_fine_faces():
        """The property that makes the figure readable, and testable.

        A label placed off its own coarse face names a connection while
        sitting on a different one.
        """
        g, _, cg = _coarse()
        pos, _ = face_number_labels(cg)
        c = np.asarray(g.faces.centroids, dtype=float)
        for f in range(cg.num_faces):
            sub = c[cg.sub_faces(f)]
            assert np.all(pos[f] >= sub.min(axis=0) - 1e-9), f
            assert np.all(pos[f] <= sub.max(axis=0) + 1e-9), f
        print("ok test_a_coarse_label_sits_among_its_fine_faces")

    def test_the_centroid_is_area_weighted():
        """MRST's private helper divides by area without weighting by it.

        On a graded grid the two answers differ, so the case is built there
        rather than on a unit Cartesian grid where they coincide.
        """
        from pymrst.gridprocessing.tensor_grid import tensor_grid
        g = compute_geometry(tensor_grid(np.array([0.0, 1.0, 5.0]),
                                         np.array([0.0, 1.0, 5.0]),
                                         np.array([0.0, 1.0])))
        p = np.zeros(g.cells.num, dtype=np.int64)
        cg = generate_coarse_grid(g, p)
        pos, _ = face_number_labels(cg)
        a = np.asarray(g.faces.areas, dtype=float)
        c = np.asarray(g.faces.centroids, dtype=float)
        differed = False
        for f in range(cg.num_faces):
            sub = cg.sub_faces(f)
            weighted = (c[sub] * a[sub, None]).sum(axis=0) / a[sub].sum()
            mrst_helper = c[sub].sum(axis=0) / a[sub].sum()
            assert np.allclose(pos[f], weighted), f
            differed |= not np.allclose(weighted, mrst_helper)
        assert differed, "the case does not distinguish the two formulas"
        print("ok test_the_centroid_is_area_weighted")

    def test_coarse_geometry_is_computed_on_demand():
        _, _, cg = _coarse()
        assert cg.face_centroids is None, "the fixture already has geometry"
        face_number_labels(cg)
        assert cg.face_centroids is not None
        print("ok test_coarse_geometry_is_computed_on_demand")

    def test_fine_geometry_is_computed_on_demand():
        g = cart_grid([2, 2, 1], [2.0, 2.0, 1.0])
        pos, labels = face_number_labels(g)
        assert pos.shape[0] == g.faces.num == len(labels)
        print("ok test_fine_geometry_is_computed_on_demand")

    def test_it_works_in_two_dimensions():
        g = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        pos, _ = face_number_labels(g)
        assert pos.shape[1] == 2, pos.shape
        assert _to_three(pos).shape[1] == 3
        print("ok test_it_works_in_two_dimensions")

    def test_plotting_is_optional():
        g = _fine()
        # An off-screen plotter is passed explicitly. These functions build a
        # `pv.Plotter` and never call `.show()`, so they do not block - but a
        # bare call is indistinguishable, to a reader or to the sweep's static
        # guard, from one that does. Being explicit costs nothing and keeps the
        # guard's rule simple: never call a drawing routine unguarded.
        try:
            import pyvista as pv
            plotter = pv.Plotter(off_screen=True)
        except ImportError:
            plotter = None
        out = plot_face_numbers(g, plotter=plotter)
        assert out is None or hasattr(out, "add_point_labels"), type(out)
        print("ok test_plotting_is_optional")

    def test_out_of_range_faces_are_rejected():
        g = _fine()
        try:
            face_number_labels(g, [g.faces.num])
        except ValueError as exc:
            assert "must lie in" in str(exc), exc
        else:
            raise AssertionError("expected an out-of-range face to fail")
        print("ok test_out_of_range_faces_are_rejected")

    test_every_face_gets_a_label_at_its_own_centroid()
    test_a_subset_keeps_its_own_numbers()
    test_labels_are_zero_based_by_default()
    test_a_coarse_grid_labels_coarse_faces()
    test_a_coarse_label_sits_among_its_fine_faces()
    test_the_centroid_is_area_weighted()
    test_coarse_geometry_is_computed_on_demand()
    test_fine_geometry_is_computed_on_demand()
    test_it_works_in_two_dimensions()
    test_plotting_is_optional()
    test_out_of_range_faces_are_rejected()
    print("\nAll tests passed!")
