"""Shift a grid in space, ported from MRST `translateGrid`."""

from dataclasses import replace

import numpy as np

from pymrst.gridprocessing.grid_structure import Grid, carry_custom_fields


def translate_grid(g: Grid, dx) -> Grid:
    """Return `g` moved by `dx`, one component per coordinate direction.

    Node coordinates move, and so does any geometry that lives in the same
    space - cell and face centroids. Areas, volumes and normals are invariant
    under translation and are left alone, so a grid that already has geometry
    keeps it.
    """
    dx = np.atleast_1d(np.asarray(dx, dtype=float)).ravel()
    coords = np.asarray(g.nodes.coords, dtype=float)
    if dx.size != coords.shape[1]:
        raise ValueError(
            f"translation needs one component per coordinate direction: "
            f"expected {coords.shape[1]}, got {dx.size}"
        )

    out = replace(
        g,
        nodes=replace(g.nodes, coords=coords + dx),
        cells=replace(
            g.cells,
            centroids=None if g.cells.centroids is None
            else np.asarray(g.cells.centroids, dtype=float) + dx,
        ),
        faces=replace(
            g.faces,
            centroids=None if g.faces.centroids is None
            else np.asarray(g.faces.centroids, dtype=float) + dx,
        ),
    )
    carry_custom_fields(g, out)
    return out


translateGrid = translate_grid


if __name__ == "__main__":
    from pymrst.gridprocessing.cart_grid import cart_grid
    from pymrst.gridprocessing.compute_geometry import compute_geometry

    def test_nodes_move_by_the_offset() -> None:
        grid = cart_grid([2, 2], [1.0, 1.0])
        before = grid.nodes.coords.copy()
        moved = translate_grid(grid, [3.0, -1.0])
        assert np.allclose(moved.nodes.coords, before + [3.0, -1.0])
        assert np.allclose(grid.nodes.coords, before), "input must not change"
        print("ok test_nodes_move_by_the_offset")

    def test_geometry_follows_and_measures_are_unchanged() -> None:
        grid = compute_geometry(cart_grid([2, 2, 2], [1.0, 1.0, 1.0]))
        moved = translate_grid(grid, [1.0, 2.0, 3.0])
        assert np.allclose(moved.cells.centroids, grid.cells.centroids + [1, 2, 3])
        assert np.allclose(moved.faces.centroids, grid.faces.centroids + [1, 2, 3])
        assert np.allclose(moved.cells.volumes, grid.cells.volumes)
        assert np.allclose(moved.faces.areas, grid.faces.areas)
        assert np.allclose(moved.faces.normals, grid.faces.normals)
        print("ok test_geometry_follows_and_measures_are_unchanged")

    def test_recomputed_geometry_agrees_with_the_shifted_one() -> None:
        grid = compute_geometry(cart_grid([3, 2], [3.0, 2.0]))
        moved = compute_geometry(translate_grid(grid, [5.0, 7.0]))
        assert np.allclose(moved.cells.centroids, grid.cells.centroids + [5.0, 7.0])
        print("ok test_recomputed_geometry_agrees_with_the_shifted_one")

    def test_grid_without_geometry_stays_without() -> None:
        moved = translate_grid(cart_grid([2, 2], [1.0, 1.0]), [1.0, 1.0])
        assert moved.cells.centroids is None and moved.faces.centroids is None
        print("ok test_grid_without_geometry_stays_without")

    def test_wrong_dimension_is_rejected() -> None:
        try:
            translate_grid(cart_grid([2, 2], [1.0, 1.0]), [1.0, 2.0, 3.0])
        except ValueError:
            print("ok test_wrong_dimension_is_rejected")
            return
        raise AssertionError("a 3-vector should be rejected for a 2D grid")

    test_nodes_move_by_the_offset()
    test_geometry_follows_and_measures_are_unchanged()
    test_recomputed_geometry_agrees_with_the_shifted_one()
    test_grid_without_geometry_stays_without()
    test_wrong_dimension_is_rejected()
    print("\nAll tests passed!")
