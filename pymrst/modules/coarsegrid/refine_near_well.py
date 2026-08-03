"""Refine a partition around a well (MRST ``refineNearWell.m``).

Flow near a well is radial and steep, so uniform blocks resolve it badly: the
pressure drop that matters happens inside the first block. This builds a
partition in polar coordinates about the well instead, splitting the points
into radial rings and angular sectors, so the blocks are small where the
gradient is large.

Rings are logarithmic by default, which is the point rather than a detail:
pressure around a well goes as `log r`, so equal-width rings in `log r` carry
roughly equal pressure drop, while equal-width rings in `r` put almost all of
it in the first one.

Distance is measured in the **xy plane only**, as in MRST. A vertical well is
the case being served, and including z would make the rings spheres around the
perforation rather than cylinders around the wellbore.

**Points outside `max_radius` get -1, not a block.** MRST marks them 0 so the
result can be added to another partition, and it uses 1-based blocks so 0 is
free. PyMRST partitions are 0-based, so 0 is a real block and the mask has to
move; -1 is the marker PyMRST already uses for "no cell" in `faces.neighbors`.
`refine_partition` performs the combination MRST's comment describes, which is
the reason the marker exists.
"""

from __future__ import annotations

import numpy as np

from pymrst.modules.coarsegrid.compress_partition import compress_partition


def refine_near_well(pts, wellpt, max_radius=np.inf, angle_bins=6,
                     radius_bins: int = 3, log_bins: bool = True):
    """Partition `pts` into rings and sectors about `wellpt`.

    Returns ``(partition, bin_angle, bin_radius, dist)``. `angle_bins` may be a
    single count or one count per radial ring. `max_radius` may be a single
    distance or a per-axis pair, which bounds a box rather than a circle.
    """
    pts = np.atleast_2d(np.asarray(pts, dtype=float))
    wellpt = np.asarray(wellpt, dtype=float).ravel()
    if wellpt.size < 2 or pts.shape[1] < 2:
        raise ValueError("points and the well need at least two coordinates")
    if int(radius_bins) < 1:
        raise ValueError(f"radius_bins must be at least 1, got {radius_bins}")

    angle_bins = np.atleast_1d(np.asarray(angle_bins, dtype=np.int64))
    if angle_bins.size not in (1, int(radius_bins)):
        raise ValueError(
            f"angle_bins must hold one value or one per radial ring "
            f"({radius_bins}), got {angle_bins.size}")
    if np.any(angle_bins < 1):
        raise ValueError("every angular bin count must be at least 1")

    n_pts = pts.shape[0]
    bin_angle = np.column_stack([
        _partition_sectors(pts, wellpt, int(a)) for a in angle_bins])
    bin_radius, dist, outside = _partition_bins(
        pts, wellpt, int(radius_bins), log_bins, max_radius)

    stride = int(bin_radius.max()) + 1
    if angle_bins.size == 1:
        partition = bin_radius + stride * bin_angle[:, 0]
    else:
        partition = np.zeros(n_pts, dtype=np.int64)
        for i in range(angle_bins.size):
            here = bin_radius == i
            partition[here] = bin_radius[here] + stride * bin_angle[here, i]

    partition = compress_partition(partition)
    partition[outside] = -1
    return partition, bin_angle, bin_radius, dist


def refine_partition(base, refinement) -> np.ndarray:
    """Split each block of `base` by `refinement`, ignoring its -1 entries.

    The combination MRST's "partitions can be created simply by addition"
    comment refers to, written out because the marker changed from 0 to -1.
    Cells marked -1 keep their base block.
    """
    base = np.asarray(base, dtype=np.int64).ravel()
    refinement = np.asarray(refinement, dtype=np.int64).ravel()
    if base.size != refinement.size:
        raise ValueError(
            f"base has {base.size} entries and the refinement {refinement.size}")
    stride = int(refinement.max()) + 2 if refinement.size else 1
    return compress_partition(base * stride + (refinement + 1))


def _partition_bins(pts, p0, n_bins, log_bins, max_radius):
    """Radial ring index, planar distance, and the out-of-range mask."""
    offset = pts[:, :2] - p0[:2]
    max_radius = np.atleast_1d(np.asarray(max_radius, dtype=float))
    if max_radius.size == 1:
        dist = np.linalg.norm(offset, axis=1)
        outside = dist > max_radius[0]
    else:
        # A per-axis limit bounds a box, so the test is per axis while the
        # distance reported stays the true planar one.
        outside = np.any(np.abs(offset) / max_radius[:2] > 1.0, axis=1)
        dist = np.linalg.norm(offset, axis=1)

    inside = ~outside
    if not np.any(inside):
        return np.zeros(pts.shape[0], dtype=np.int64), dist, outside

    d = _normalize(dist)
    if log_bins:
        # log10(10x + 1) maps [0, 1] onto itself while stretching the near-well
        # end, which is where the pressure drop is.
        d = np.log10(d * 10.0 + 1.0)
    d = d.copy()
    d[outside] = d[inside].max()
    return _group_by_bins(d, n_bins), dist, outside


def _partition_sectors(pts, p0, n_sectors) -> np.ndarray:
    return _group_by_bins(_angle(p0, pts), n_sectors, -np.pi, np.pi)


def _angle(p0, p1) -> np.ndarray:
    """Bearing of each point from the well, in the xy plane.

    MRST writes `atan2(v(1), v(2))`, which measures from the y axis rather than
    the x axis. That only rotates the sector boundaries, and the port keeps it
    so the sectors land where MATLAB puts them.
    """
    v = p1[:, :2] - p0[:2]
    return np.arctan2(v[:, 0], v[:, 1])


def _group_by_bins(data, n_bins, d_min=None, d_max=None) -> np.ndarray:
    """Equal-width bins over `[d_min, d_max]`, as 0-based indices.

    The lower edge belongs to bin 0 and each bin is then half-open on the left,
    matching MRST's `data > (i-1)*w & data <= i*w` with a default of 1.
    """
    data = np.asarray(data, dtype=float).ravel()
    if n_bins == 1:
        return np.zeros(data.size, dtype=np.int64)
    lo = float(np.min(data)) if d_min is None else float(d_min)
    hi = float(np.max(data)) if d_max is None else float(d_max)
    if hi <= lo:
        return np.zeros(data.size, dtype=np.int64)

    scaled = (data - lo) / (hi - lo)
    out = np.zeros(data.size, dtype=np.int64)
    width = 1.0 / n_bins
    for i in range(1, n_bins + 1):
        out[(scaled > (i - 1) * width) & (scaled <= i * width)] = i - 1
    return out


def _normalize(data: np.ndarray) -> np.ndarray:
    shifted = data - data.min()
    largest = shifted.max()
    return shifted if largest <= 0.0 else shifted / largest


__all__ = ["refine_near_well", "refine_partition"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.partition_cart_grid import partition_cart_grid

    def _case(dims=(20, 20, 1)):
        g = compute_geometry(cart_grid(list(dims), [20.0, 20.0, 1.0]))
        return g, np.asarray(g.cells.centroids, dtype=float), np.array([10.0, 10.0, 0.5])

    def test_the_partition_covers_every_point():
        _, pts, well = _case()
        p, _, _, _ = refine_near_well(pts, well)
        assert p.size == pts.shape[0]
        assert np.all(p >= 0), "nothing should be masked without a max_radius"
        print(f"ok test_the_partition_covers_every_point ({p.max() + 1} blocks)")

    def test_blocks_are_rings_crossed_with_sectors():
        """The defining structure: a block is one ring and one sector."""
        _, pts, well = _case()
        p, angle, radius, _ = refine_near_well(pts, well, angle_bins=4,
                                               radius_bins=3)
        for b in np.unique(p):
            here = p == b
            assert len(set(radius[here].tolist())) == 1, (b, "ring")
            assert len(set(angle[here, 0].tolist())) == 1, (b, "sector")
        assert p.max() + 1 <= 12, p.max() + 1
        print(f"ok test_blocks_are_rings_crossed_with_sectors "
              f"({p.max() + 1} of 12 possible)")

    def test_rings_grow_outward():
        """Ring index must increase with distance, or the refinement inverts.

        A reversed ring order still produces the right number of blocks of the
        right shapes, and puts the fine blocks at the far boundary.
        """
        _, pts, well = _case()
        _, _, radius, dist = refine_near_well(pts, well, radius_bins=4)
        for a in range(int(radius.max())):
            inner = dist[radius == a]
            outer = dist[radius == a + 1]
            if inner.size and outer.size:
                assert inner.max() <= outer.min() + 1e-9, (a, inner.max(),
                                                           outer.min())
        print("ok test_rings_grow_outward")

    def test_log_bins_put_more_resolution_near_the_well():
        """The reason logarithmic rings are the default.

        With linear rings the innermost one is as wide as the others and
        swallows the whole near-well pressure drop. Both give a valid
        partition; only the widths differ.
        """
        _, pts, well = _case()
        _, _, log_r, dist = refine_near_well(pts, well, radius_bins=4,
                                             log_bins=True)
        _, _, lin_r, _ = refine_near_well(pts, well, radius_bins=4,
                                          log_bins=False)
        log_first = dist[log_r == 0].max()
        lin_first = dist[lin_r == 0].max()
        assert log_first < lin_first, (log_first, lin_first)
        print(f"ok test_log_bins_put_more_resolution_near_the_well "
              f"(inner ring {log_first:.2f} m vs {lin_first:.2f} m)")

    def test_sectors_divide_the_full_circle():
        """Bins run over -pi to pi regardless of where the points fall.

        Taking the range from the data instead would rotate the sectors with
        the sample and give unequal ones.
        """
        _, pts, well = _case()
        _, angle, _, _ = refine_near_well(pts, well, angle_bins=4, radius_bins=1)
        counts = np.bincount(angle[:, 0], minlength=4)
        assert counts.min() > 0, counts
        # A symmetric point cloud about the well fills the sectors evenly.
        assert counts.max() - counts.min() <= 0.25 * counts.mean(), counts
        print(f"ok test_sectors_divide_the_full_circle ({counts.tolist()})")

    def test_max_radius_masks_the_far_field():
        _, pts, well = _case()
        p, _, _, dist = refine_near_well(pts, well, max_radius=4.0)
        assert np.all(p[dist > 4.0] == -1), "a far point kept a block"
        assert np.all(p[dist <= 4.0] >= 0), "a near point was masked"
        print(f"ok test_max_radius_masks_the_far_field "
              f"({int(np.count_nonzero(p < 0))} of {p.size} masked)")

    def test_a_per_axis_radius_bounds_a_box():
        _, pts, well = _case()
        p, _, _, _ = refine_near_well(pts, well, max_radius=np.array([3.0, 8.0]))
        offset = np.abs(pts[:, :2] - well[:2])
        expected = (offset[:, 0] > 3.0) | (offset[:, 1] > 8.0)
        assert np.array_equal(p < 0, expected)
        print("ok test_a_per_axis_radius_bounds_a_box")

    def test_distance_ignores_depth():
        """Cylinders about the wellbore, not spheres about the perforation."""
        g = compute_geometry(cart_grid([4, 4, 4], [4.0, 4.0, 40.0]))
        pts = np.asarray(g.cells.centroids, dtype=float)
        well = np.array([2.0, 2.0, 20.0])
        _, _, _, dist = refine_near_well(pts, well)
        planar = np.linalg.norm(pts[:, :2] - well[:2], axis=1)
        assert np.allclose(dist, planar), "depth leaked into the distance"
        print("ok test_distance_ignores_depth")

    def test_one_bin_each_way_gives_one_block():
        _, pts, well = _case()
        p, _, _, _ = refine_near_well(pts, well, angle_bins=1, radius_bins=1)
        assert p.max() + 1 == 1, p.max() + 1
        print("ok test_one_bin_each_way_gives_one_block")

    def test_angle_bins_may_vary_by_ring():
        """Fewer sectors far out, where the flow is no longer radial."""
        _, pts, well = _case()
        p, angle, radius, _ = refine_near_well(pts, well, angle_bins=[8, 4, 1],
                                               radius_bins=3)
        assert angle.shape[1] == 3, angle.shape
        for ring, expected in enumerate([8, 4, 1]):
            here = radius == ring
            assert len(set(p[here].tolist())) <= expected, (ring, expected)
        print("ok test_angle_bins_may_vary_by_ring")

    def test_the_partition_is_contiguous():
        _, pts, well = _case()
        p, _, _, _ = refine_near_well(pts, well)
        assert np.array_equal(np.unique(p), np.arange(p.max() + 1))
        print("ok test_the_partition_is_contiguous")

    def test_refine_partition_splits_without_crossing_blocks():
        """Combining must refine the base, never merge across it.

        A careless combination would let two cells from different base blocks
        share a block, which is the one thing the operation must not do.
        """
        g, pts, well = _case()
        base = partition_cart_grid([20, 20, 1], [2, 2, 1])
        refinement, _, _, _ = refine_near_well(pts, well, max_radius=5.0)
        out = refine_partition(base, refinement)
        for b in np.unique(out):
            assert len(set(base[out == b].tolist())) == 1, b
        assert out.max() > base.max(), (out.max(), base.max())
        print(f"ok test_refine_partition_splits_without_crossing_blocks "
              f"({base.max() + 1} blocks became {out.max() + 1})")

    def test_masked_cells_keep_their_base_block():
        g, pts, well = _case()
        base = partition_cart_grid([20, 20, 1], [2, 2, 1])
        refinement, _, _, _ = refine_near_well(pts, well, max_radius=5.0)
        out = refine_partition(base, refinement)
        far = refinement < 0
        for b in np.unique(base):
            here = far & (base == b)
            if np.any(here):
                assert len(set(out[here].tolist())) == 1, b
        print("ok test_masked_cells_keep_their_base_block")

    def test_bad_input_is_rejected():
        _, pts, well = _case()
        for kwargs, why in (({"radius_bins": 0}, "no radial bins"),
                            ({"angle_bins": [2, 3], "radius_bins": 3},
                             "wrong number of angular bins"),
                            ({"angle_bins": 0}, "no angular bins")):
            try:
                refine_near_well(pts, well, **kwargs)
            except ValueError:
                pass
            else:
                raise AssertionError(f"expected {why} to be rejected")
        try:
            refine_near_well(pts[:, :1], well)
        except ValueError as exc:
            assert "two coordinates" in str(exc), exc
        else:
            raise AssertionError("expected one-dimensional points to be rejected")
        print("ok test_bad_input_is_rejected")

    test_the_partition_covers_every_point()
    test_blocks_are_rings_crossed_with_sectors()
    test_rings_grow_outward()
    test_log_bins_put_more_resolution_near_the_well()
    test_sectors_divide_the_full_circle()
    test_max_radius_masks_the_far_field()
    test_a_per_axis_radius_bounds_a_box()
    test_distance_ignores_depth()
    test_one_bin_each_way_gives_one_block()
    test_angle_bins_may_vary_by_ring()
    test_the_partition_is_contiguous()
    test_refine_partition_splits_without_crossing_blocks()
    test_masked_cells_keep_their_base_block()
    test_bad_input_is_rejected()
    print("\nAll tests passed!")
