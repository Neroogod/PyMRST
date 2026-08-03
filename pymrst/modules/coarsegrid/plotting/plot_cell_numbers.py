"""Label cells or blocks with their numbers (MRST ``plotCellNumbers.m``).

Draws each cell's index at its centroid. On a coarse grid it labels blocks
instead, which is how MRST's coarse-grid examples show the correspondence
between a fine grid and its partition.

Positions are computed by `cell_number_labels` and drawn by `plot_cell_numbers`,
so the placement is testable without a display. A label in the wrong place is a
picture that reads correctly and says something false, which no rendering test
would catch.

**Numbers are 0-based here and 1-based in MATLAB.** PyMRST indexes from 0
throughout, and a figure whose labels disagree with the indices the user is
typing is worse than one that differs from MATLAB. `base=1` reproduces MATLAB's
labels for a side-by-side comparison.

**MRST's private `computeCoarseGeometry` is not reproduced.** Both numbering
plots carry a private copy of it, and both compute coarse face centroids as the
plain sum of the fine centroids divided by the *total area* rather than as an
area-weighted mean, which is only correct when every fine face has unit area.
`plotCellNumbers`'s copy also writes the result to a misspelled field
(`centriods`), so its face centroids stay zero and never reach a plot. The port
calls `coarsen_geometry` instead, which is MRST's own correct implementation of
the same quantity.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np


def cell_number_labels(g, cells: Sequence[int] | None = None, base: int = 0):
    """Return ``(positions, labels)`` for the given cells or blocks.

    `g` may be a grid or a `CoarseGrid`. Geometry is computed if it is missing.
    """
    centroids = _cell_centroids(g)
    n = centroids.shape[0]
    if cells is None:
        cells = np.arange(n, dtype=np.int64)
    cells = np.asarray(cells, dtype=np.int64).ravel()
    if cells.size and (cells.min() < 0 or cells.max() >= n):
        raise ValueError(f"cell numbers must lie in 0 to {n - 1}")
    return centroids[cells], [str(int(c) + base) for c in cells]


def _cell_centroids(g) -> np.ndarray:
    """Centroids of `g`'s cells, or of a coarse grid's blocks."""
    if hasattr(g, "partition") and hasattr(g, "n_blocks"):
        if getattr(g, "cell_centroids", None) is None:
            from pymrst.modules.coarsegrid.utils.coarsen_geometry import (
                coarsen_geometry,
            )
            coarsen_geometry(g)
        return np.asarray(g.cell_centroids, dtype=float)
    if getattr(g.cells, "centroids", None) is None:
        from pymrst.gridprocessing import compute_geometry
        g = compute_geometry(g)
    return np.asarray(g.cells.centroids, dtype=float)


def plot_cell_numbers(g, cells: Sequence[int] | None = None, base: int = 0,
                      plotter=None, **kwargs) -> Any:
    """Draw the labels, returning the plotter.

    Returns ``None`` when no plotting backend is installed, so the positions
    above stay usable headless. `PlotResult` is not used because it carries the
    face geometry behind a patch, and text labels have none.
    """
    positions, labels = cell_number_labels(g, cells, base)
    try:
        import pyvista as pv
    except ImportError:
        return None
    plotter = plotter or pv.Plotter()
    plotter.add_point_labels(_to_three(positions), labels, **kwargs)
    return plotter


def _to_three(points: np.ndarray) -> np.ndarray:
    """Pad 2D positions with a zero z, which the 3D renderer needs."""
    points = np.atleast_2d(points)
    if points.shape[1] >= 3:
        return points[:, :3]
    return np.column_stack([points, np.zeros((points.shape[0], 3 - points.shape[1]))])


__all__ = ["cell_number_labels", "plot_cell_numbers"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
    from pymrst.modules.coarsegrid.partition_cart_grid import partition_cart_grid

    def _fine(dims=(3, 2, 1)):
        return compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))

    def test_every_cell_gets_a_label_at_its_own_centroid():
        """The property a picture cannot confirm.

        Labels offset by one still produce a tidy figure, and every number in
        it names the wrong cell.
        """
        g = _fine()
        pos, labels = cell_number_labels(g)
        assert len(labels) == g.cells.num
        for c in range(g.cells.num):
            assert labels[c] == str(c)
            assert np.allclose(pos[c], np.asarray(g.cells.centroids)[c])
        print(f"ok test_every_cell_gets_a_label_at_its_own_centroid "
              f"({g.cells.num} cells)")

    def test_a_subset_keeps_its_own_numbers():
        g = _fine()
        pos, labels = cell_number_labels(g, [4, 1])
        assert labels == ["4", "1"], labels
        assert np.allclose(pos[0], np.asarray(g.cells.centroids)[4])
        print("ok test_a_subset_keeps_its_own_numbers")

    def test_labels_are_zero_based_by_default():
        g = _fine()
        assert cell_number_labels(g)[1][0] == "0"
        assert cell_number_labels(g, base=1)[1][0] == "1"
        print("ok test_labels_are_zero_based_by_default")

    def test_a_coarse_grid_labels_blocks_not_cells():
        """The case MRST wrote the function for."""
        g = _fine((4, 4, 1))
        p = partition_cart_grid([4, 4, 1], [2, 2, 1])
        cg = generate_coarse_grid(g, p)
        pos, labels = cell_number_labels(cg)
        assert len(labels) == cg.n_blocks == 4, labels
        assert pos.shape[0] == 4
        print(f"ok test_a_coarse_grid_labels_blocks_not_cells "
              f"({cg.n_blocks} blocks, {g.cells.num} cells)")

    def test_a_block_label_sits_inside_its_block():
        g = _fine((4, 4, 1))
        p = partition_cart_grid([4, 4, 1], [2, 2, 1])
        cg = generate_coarse_grid(g, p)
        pos, _ = cell_number_labels(cg)
        c = np.asarray(g.cells.centroids, dtype=float)
        for b in range(cg.n_blocks):
            inside = c[p == b]
            assert np.all(pos[b] >= inside.min(axis=0) - 1e-9), b
            assert np.all(pos[b] <= inside.max(axis=0) + 1e-9), b
        print("ok test_a_block_label_sits_inside_its_block")

    def test_coarse_geometry_is_computed_on_demand():
        g = _fine((4, 4, 1))
        cg = generate_coarse_grid(g, partition_cart_grid([4, 4, 1], [2, 2, 1]))
        assert cg.cell_centroids is None, "the fixture already has geometry"
        cell_number_labels(cg)
        assert cg.cell_centroids is not None, "geometry was not filled in"
        print("ok test_coarse_geometry_is_computed_on_demand")

    def test_fine_geometry_is_computed_on_demand():
        g = cart_grid([2, 2, 1], [2.0, 2.0, 1.0])       # no compute_geometry
        pos, labels = cell_number_labels(g)
        assert pos.shape[0] == 4 and len(labels) == 4
        print("ok test_fine_geometry_is_computed_on_demand")

    def test_two_dimensional_positions_are_padded_for_the_renderer():
        g = compute_geometry(cart_grid([2, 2], [2.0, 2.0]))
        pos, _ = cell_number_labels(g)
        assert pos.shape[1] == 2, pos.shape
        assert _to_three(pos).shape == (4, 3)
        assert np.all(_to_three(pos)[:, 2] == 0.0)
        print("ok test_two_dimensional_positions_are_padded_for_the_renderer")

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
        out = plot_cell_numbers(g, plotter=plotter)
        assert out is None or hasattr(out, "add_point_labels"), type(out)
        print("ok test_plotting_is_optional")

    def test_out_of_range_cells_are_rejected():
        g = _fine()
        try:
            cell_number_labels(g, [g.cells.num])
        except ValueError as exc:
            assert "must lie in" in str(exc), exc
        else:
            raise AssertionError("expected an out-of-range cell to fail")
        print("ok test_out_of_range_cells_are_rejected")

    test_every_cell_gets_a_label_at_its_own_centroid()
    test_a_subset_keeps_its_own_numbers()
    test_labels_are_zero_based_by_default()
    test_a_coarse_grid_labels_blocks_not_cells()
    test_a_block_label_sits_inside_its_block()
    test_coarse_geometry_is_computed_on_demand()
    test_fine_geometry_is_computed_on_demand()
    test_two_dimensional_positions_are_padded_for_the_renderer()
    test_plotting_is_optional()
    test_out_of_range_cells_are_rejected()
    print("\nAll tests passed!")
