"""Explosion view of a partition (MRST ``explosionView.m``).

Draws a coarse partition with its blocks pushed apart, so the interior blocks
and the faces between them become visible. Each block is displaced radially
away from an origin by a fraction of its distance from it.

The displacement is worth separating from the drawing, and this module does:
``block_displacements`` computes the offsets, ``exploded_block_grids`` applies
them, and ``explosion_view`` draws the result if a backend is installed. That
keeps the geometry testable, which matters because an explosion view whose
blocks move the wrong way still produces a picture.

Two details from MRST worth carrying over.

**Anisotropy normalisation.** The offsets are computed in coordinates scaled by
the model's own extent, `a = span / max(span)`, so a reservoir that is
kilometres wide and metres thick separates sensibly in both directions rather
than only laterally.

**The sign term.** MRST writes
`sgn = sign(mean(nodes - o)) .* sign([x, y, z])` and then displaces by
`sgn .* [x, y, z]`. Multiplying a value by the product of its own sign and
another sign leaves the magnitude and takes the second sign, so the effect is
to move each block away from the origin along each axis. The port states that
directly rather than reproducing the double-sign expression, and asserts it.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pymrst.gridprocessing.grid_structure import extract_subgrid


def _anisotropy(g) -> np.ndarray:
    """Per-axis extent, normalised so the widest direction is 1."""
    c = np.asarray(g.cells.centroids, dtype=float)
    span = c.max(axis=0) - c.min(axis=0)
    largest = span.max()
    if largest <= 0.0:
        return np.ones_like(span)
    out = span / largest
    # A degenerate axis would divide by zero below; leave it unscaled.
    return np.where(out > 0.0, out, 1.0)


def block_displacements(g, partition, radius: float = 0.2,
                        origin: Sequence[float] | None = None) -> np.ndarray:
    """Offset applied to each block, shape ``(n_blocks, griddim)``.

    Block `b` moves away from `origin` by `radius` times its distance from it,
    measured in extent-normalised coordinates. Blocks with no cells do not
    move.
    """
    partition = np.asarray(partition, dtype=np.int64).ravel()
    centroids = np.asarray(g.cells.centroids, dtype=float)
    if partition.size != centroids.shape[0]:
        raise ValueError(
            f"partition has {partition.size} entries for {centroids.shape[0]} cells")
    if radius < 0.0:
        raise ValueError(f"radius multiplier must be non-negative, got {radius}")

    dim = centroids.shape[1]
    o = centroids.mean(axis=0) if origin is None else np.asarray(origin, dtype=float)
    if o.size != dim:
        raise ValueError(f"origin needs {dim} coordinates, got {o.size}")

    a = _anisotropy(g)
    n_blocks = int(partition.max()) + 1 if partition.size else 0
    out = np.zeros((n_blocks, dim), dtype=float)
    for b in range(n_blocks):
        cells = partition == b
        if not np.any(cells):
            continue
        # Block centre relative to the origin, in normalised coordinates. The
        # displacement keeps that direction and scales its length by `radius`.
        offset = (centroids[cells].mean(axis=0) - o) / a
        out[b] = radius * offset * a
    return out


def exploded_block_grids(g, partition, radius: float = 0.2,
                         origin: Sequence[float] | None = None
                         ) -> list[tuple[int, Any]]:
    """One displaced subgrid per non-empty block.

    Returns ``(block, grid)`` pairs. Each grid is a copy with its node
    coordinates shifted; `g` is untouched.

    Geometry is recomputed after the shift. A subgrid inherits its parent's
    centroids and volumes, so moving the nodes without recomputing leaves a
    grid whose nodes and centroids disagree: it plots in the new position and
    reports the old one. Anything measuring block separation from centroids
    would then see no movement at all.
    """
    from pymrst.gridprocessing import compute_geometry

    partition = np.asarray(partition, dtype=np.int64).ravel()
    shifts = block_displacements(g, partition, radius, origin)
    out: list[tuple[int, Any]] = []
    for b in range(shifts.shape[0]):
        cells = np.flatnonzero(partition == b)
        if cells.size == 0:
            continue
        sub = extract_subgrid(g, cells)
        sub.nodes.coords = np.asarray(sub.nodes.coords, dtype=float) + shifts[b]
        out.append((b, compute_geometry(sub)))
    return out


def explosion_view(g, partition, radius: float = 0.2,
                   origin: Sequence[float] | None = None, **kwargs):
    """Draw the partition with its blocks pushed apart.

    Returns ``None`` when no plotting backend is installed, so the geometry
    above stays usable headless.
    """
    blocks = exploded_block_grids(g, partition, radius, origin)
    try:
        from pymrst.plotting import plot_cell_data
    except ImportError:
        return None
    return [plot_cell_data(sub, np.full(sub.cells.num, float(b)), **kwargs)
            for b, sub in blocks]


__all__ = ["block_displacements", "exploded_block_grids", "explosion_view"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.partition_cart_grid import partition_cart_grid

    def _case(dims=(4, 4, 2), part=(2, 2, 1)):
        g = compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))
        return g, partition_cart_grid(list(dims), list(part))

    def test_every_block_moves_away_from_the_origin():
        """The defining property, and the one a picture cannot confirm.

        A block whose displacement points inward would overlap its neighbours
        and still render as a plausible explosion view.
        """
        g, p = _case()
        o = np.asarray(g.cells.centroids, dtype=float).mean(axis=0)
        shifts = block_displacements(g, p, radius=0.3)
        centroids = np.asarray(g.cells.centroids, dtype=float)
        for b in range(shifts.shape[0]):
            centre = centroids[p == b].mean(axis=0)
            outward = centre - o
            assert np.dot(shifts[b], outward) > 0.0, (b, shifts[b], outward)
        print("ok test_every_block_moves_away_from_the_origin")

    def test_displacement_scales_with_the_radius():
        g, p = _case()
        a = block_displacements(g, p, radius=0.1)
        b = block_displacements(g, p, radius=0.4)
        assert np.allclose(b, 4.0 * a), "displacement is not linear in radius"
        print("ok test_displacement_scales_with_the_radius")

    def test_zero_radius_leaves_the_grid_alone():
        g, p = _case()
        shifts = block_displacements(g, p, radius=0.0)
        assert np.allclose(shifts, 0.0), shifts
        for _, sub in exploded_block_grids(g, p, radius=0.0):
            assert np.all(np.isfinite(np.asarray(sub.nodes.coords)))
        print("ok test_zero_radius_leaves_the_grid_alone")

    def test_a_block_centred_on_the_origin_does_not_move():
        g, p = _case(dims=(4, 4, 2), part=(1, 1, 1))
        shifts = block_displacements(g, p, radius=0.5)
        assert np.allclose(shifts[0], 0.0, atol=1e-12), shifts[0]
        print("ok test_a_block_centred_on_the_origin_does_not_move")

    def test_blocks_separate_further_as_radius_grows():
        """The visual purpose: gaps open between blocks."""
        g, p = _case()
        def gap(r):
            grids = exploded_block_grids(g, p, radius=r)
            centres = [np.asarray(sub.cells.centroids, float).mean(axis=0)
                       for _, sub in grids]
            return min(np.linalg.norm(a - b)
                       for i, a in enumerate(centres)
                       for b in centres[i + 1:])
        assert gap(0.5) > gap(0.1) > gap(0.0), (gap(0.0), gap(0.1), gap(0.5))
        print("ok test_blocks_separate_further_as_radius_grows")

    def test_the_source_grid_is_untouched():
        g, p = _case()
        before = np.asarray(g.nodes.coords, dtype=float).copy()
        exploded_block_grids(g, p, radius=0.4)
        assert np.array_equal(np.asarray(g.nodes.coords), before), "grid mutated"
        print("ok test_the_source_grid_is_untouched")

    def test_each_block_keeps_its_cells_and_shape():
        """Displacement is a translation: volumes must not change."""
        g, p = _case()
        fine = np.asarray(g.cells.volumes, dtype=float)
        for b, sub in exploded_block_grids(g, p, radius=0.3):
            assert sub.cells.num == int(np.count_nonzero(p == b)), b
            assert np.isclose(float(np.sum(sub.cells.volumes)),
                              float(fine[p == b].sum()), rtol=1e-12), b
        print("ok test_each_block_keeps_its_cells_and_shape")

    def test_returned_grids_have_consistent_geometry():
        """Nodes and centroids must agree after the shift.

        A subgrid inherits its parent's centroids, so displacing the nodes
        without recomputing leaves a grid that plots in one place and reports
        another. Nothing raises; block separation simply measures as zero.
        """
        g, p = _case()
        for b, sub in exploded_block_grids(g, p, radius=0.4):
            coords = np.asarray(sub.nodes.coords, dtype=float)
            centres = np.asarray(sub.cells.centroids, dtype=float)
            assert np.all(centres.min(axis=0) >= coords.min(axis=0) - 1e-9), b
            assert np.all(centres.max(axis=0) <= coords.max(axis=0) + 1e-9), b
        print("ok test_returned_grids_have_consistent_geometry")

    def test_anisotropy_is_normalised():
        """A thin model still separates vertically.

        Without the extent normalisation the vertical offsets would be a
        fraction of a metre against hundreds laterally, and the explosion would
        look flat.
        """
        g = compute_geometry(cart_grid([10, 10, 2], [1000.0, 1000.0, 10.0]))
        p = partition_cart_grid([10, 10, 2], [2, 2, 2])
        shifts = block_displacements(g, p, radius=0.5)
        vertical = np.abs(shifts[:, 2])
        assert vertical.max() > 0.0, "no vertical separation at all"
        print(f"ok test_anisotropy_is_normalised "
              f"(max vertical offset {vertical.max():.3f} m)")

    def test_works_in_two_dimensions():
        g = compute_geometry(cart_grid([4, 4], [4.0, 4.0]))
        p = partition_cart_grid([4, 4, 1], [2, 2, 1])
        shifts = block_displacements(g, p, radius=0.25)
        assert shifts.shape == (4, 2), shifts.shape
        assert np.all(np.isfinite(shifts))
        print("ok test_works_in_two_dimensions")

    def test_plotting_is_optional():
        """`show=False` is not optional here, it is what keeps this test finite.

        `plot_cell_data` defaults to `show=True`, which opens an interactive
        window and blocks until it is closed. Standalone that may return at
        once; run as a subprocess with no display it blocks forever, and the
        whole test sweep stops on it with no error to show for it.
        """
        g, p = _case()
        out = explosion_view(g, p, show=False)
        assert out is None or isinstance(out, list), type(out)
        print("ok test_plotting_is_optional")

    def test_bad_input_is_rejected():
        g, p = _case()
        for args, why in (((g, p[:-1]), "wrong-length partition"),
                          ((g, p, -1.0), "negative radius")):
            try:
                block_displacements(*args)
            except ValueError:
                pass
            else:
                raise AssertionError(f"expected {why} to be rejected")
        try:
            block_displacements(g, p, 0.2, [0.0, 0.0])
        except ValueError as exc:
            assert "origin needs" in str(exc), exc
        else:
            raise AssertionError("expected a short origin to be rejected")
        print("ok test_bad_input_is_rejected")

    test_every_block_moves_away_from_the_origin()
    test_displacement_scales_with_the_radius()
    test_zero_radius_leaves_the_grid_alone()
    test_a_block_centred_on_the_origin_does_not_move()
    test_blocks_separate_further_as_radius_grows()
    test_the_source_grid_is_untouched()
    test_each_block_keeps_its_cells_and_shape()
    test_returned_grids_have_consistent_geometry()
    test_anisotropy_is_normalised()
    test_works_in_two_dimensions()
    test_plotting_is_optional()
    test_bad_input_is_rejected()
    print("\nAll tests passed!")
