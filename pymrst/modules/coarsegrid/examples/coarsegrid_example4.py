"""More About Partition Vectors (``coarsegridExample4.m``).

A partition is just one block number per cell. Nothing requires it to come from
logical indices, and this shows three ways of building one that do not:

* **from polar coordinates** — angular sectors plus a disc at the centre;
* **from the sign of a function** — two blocks, each of which is several
  disconnected pieces until `process_partition` separates them;
* **by combining two partitions** — the common intersection operation, done two
  ways that must agree.

The combination is worth stating precisely because MRST shows both forms and
times them. `unique(..., 'rows')` on the pair `[pf, pc]` and the arithmetic
`compress_partition(pf + max(pf) * pc)` give the same *grouping*; they do not
generally give the same numbering. Both are here, and `same_grouping` checks
they agree on what matters.

The polar case is the one with a trap in it. Sectors are defined by angle, so
the sector containing the negative x-axis wraps from +pi to -pi and is a single
block number covering two pieces that meet only at the origin. That is exactly
what `process_partition` is for, and the tests assert the block count before and
after.
"""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.coarsegrid.compress_partition import compress_partition
from pymrst.modules.coarsegrid.generate_coarse_grid import generate_coarse_grid
from pymrst.modules.coarsegrid.partition_cart_grid import partition_cart_grid
from pymrst.modules.coarsegrid.process_partition import process_partition


def polar_partition(dims=(11, 11), radius: float = 0.3):
    """Four angular sectors plus a central disc, on a grid centred at origin.

    Returns ``(g, before, after)``, the partition as the angle gives it and
    after `process_partition` has separated the pieces that do not touch.
    """
    g = cart_grid(list(dims), [2.0, 2.0])
    g.nodes.coords = np.asarray(g.nodes.coords, dtype=float) - 1.0
    g = compute_geometry(g)
    c = np.asarray(g.cells.centroids, dtype=float)

    theta = np.arctan2(c[:, 1], c[:, 0])
    r = np.hypot(c[:, 0], c[:, 1])
    before = np.mod(np.round(4.0 * theta / np.pi).astype(np.int64) + 4, 4)
    before[r < radius] = int(before.max()) + 1
    before = compress_partition(before)
    return g, before, process_partition(g, before)


def sign_partition(dims=(20, 20), nx: int = 3, ny: int = 3):
    """Two blocks from the sign of a product of sines, then split.

    Returns ``(g, values, before, after, cg)``. `values` is the function itself,
    which MRST surfaces beside the partition.
    """
    g = compute_geometry(cart_grid(list(dims), [1.0, 1.0]))
    c = np.asarray(g.cells.centroids, dtype=float)
    values = np.sin(np.pi * nx * c[:, 0]) * np.sin(np.pi * ny * c[:, 1])
    before = (values > 0.0).astype(np.int64)
    after = process_partition(g, before)
    return g, values, before, after, generate_coarse_grid(g, after)


def combined_partitions(dims=(20, 20), blocks=(4, 4), waves: float = 4.0):
    """Intersect a stripe indicator with a Cartesian partition, two ways.

    Returns ``(g, stripes, cartesian, by_unique, by_arithmetic)``. The last two
    are the same partition under different numbering.
    """
    g = compute_geometry(cart_grid(list(dims), [1.0, 1.0]))
    c = np.asarray(g.cells.centroids, dtype=float)
    stripes = (np.sin(waves * np.pi * (c[:, 0] - c[:, 1])) > 0.0).astype(np.int64)
    cartesian = partition_cart_grid(list(dims) + [1], list(blocks) + [1])

    _, by_unique = np.unique(np.column_stack([stripes, cartesian]), axis=0,
                             return_inverse=True)
    by_unique = by_unique.astype(np.int64).ravel()
    stride = int(stripes.max()) + 1
    by_arithmetic = compress_partition(stripes + stride * cartesian)
    return g, stripes, cartesian, by_unique, by_arithmetic


__all__ = ["combined_partitions", "polar_partition", "sign_partition"]


if __name__ == "__main__":
    from pymrst.modules.coarsegrid.utils.is_same_partition import same_grouping

    def test_the_polar_partition_has_sectors_and_a_disc():
        g, before, _ = polar_partition()
        c = np.asarray(g.cells.centroids, dtype=float)
        r = np.hypot(c[:, 0], c[:, 1])
        centre = before[r < 0.3]
        assert centre.size > 0 and len(set(centre.tolist())) == 1
        assert before.max() + 1 == 5, before.max() + 1
        print(f"ok test_the_polar_partition_has_sectors_and_a_disc "
              f"({before.max() + 1} blocks, {centre.size} cells in the disc)")

    def test_the_wrapping_sector_is_disconnected_until_processed():
        """The trap in defining blocks by angle.

        The sector straddling the negative x-axis wraps from +pi to -pi, so one
        block number covers two pieces that touch only at the origin. Nothing
        about the partition vector reveals it.
        """
        _, before, after = polar_partition()
        assert after.max() > before.max(), (before.max(), after.max())
        print(f"ok test_the_wrapping_sector_is_disconnected_until_processed "
              f"({before.max() + 1} blocks became {after.max() + 1})")

    def test_processing_the_polar_partition_only_splits():
        _, before, after = polar_partition()
        assert same_grouping(after, after)
        for a in range(after.size):
            for b in range(a + 1, after.size):
                if after[a] == after[b]:
                    assert before[a] == before[b], (a, b)
        print("ok test_processing_the_polar_partition_only_splits")

    def test_every_polar_block_is_connected_after_processing():
        g, _, after = polar_partition()
        assert np.array_equal(process_partition(g, after), after)
        print("ok test_every_polar_block_is_connected_after_processing")

    def test_a_partition_needs_no_logical_indices():
        """Two blocks straight from the sign of a function."""
        g, values, before, after, cg = sign_partition()
        assert before.max() + 1 == 2, before.max() + 1
        assert np.all((values > 0) == (before == 1))
        print(f"ok test_a_partition_needs_no_logical_indices "
              f"({int(np.count_nonzero(before))} cells positive)")

    def test_the_sign_partition_is_many_disconnected_patches():
        """A 3x3 pattern of sine lobes gives 9 patches per sign, not 2 blocks.

        Left unsplit, the coarse grid would have two blocks each spread across
        the whole model, and flow would pass between lobes that do not touch.
        """
        g, _, before, after, cg = sign_partition()
        assert after.max() + 1 > 2
        assert cg.n_blocks == after.max() + 1
        assert np.array_equal(process_partition(g, after), after)
        print(f"ok test_the_sign_partition_is_many_disconnected_patches "
              f"(2 blocks became {after.max() + 1})")

    def test_the_two_combination_methods_agree():
        """MRST shows both and times them; they must describe one partition.

        The numbering differs, so an elementwise comparison would report a
        difference that is not there. Only the grouping is meaningful.
        """
        _, _, _, by_unique, by_arithmetic = combined_partitions()
        assert same_grouping(by_unique, by_arithmetic)
        assert by_unique.max() == by_arithmetic.max()
        print(f"ok test_the_two_combination_methods_agree "
              f"({by_unique.max() + 1} blocks both ways)")

    def test_the_combination_refines_both_inputs():
        """Every block must lie inside one stripe and one Cartesian block.

        A combination that crossed either would merge cells the user asked to
        keep apart, and the block count alone would not show it.
        """
        _, stripes, cartesian, by_unique, _ = combined_partitions()
        for b in np.unique(by_unique):
            here = by_unique == b
            assert len(set(stripes[here].tolist())) == 1, b
            assert len(set(cartesian[here].tolist())) == 1, b
        print("ok test_the_combination_refines_both_inputs")

    def test_the_combination_is_finer_than_either_input():
        _, stripes, cartesian, by_unique, _ = combined_partitions()
        assert by_unique.max() + 1 > max(stripes.max(), cartesian.max()) + 1
        print(f"ok test_the_combination_is_finer_than_either_input "
              f"({stripes.max() + 1} x {cartesian.max() + 1} inputs gave "
              f"{by_unique.max() + 1} blocks)")

    def test_the_arithmetic_form_needs_the_right_stride():
        """`pf + max(pf) * pc` only works because the stride exceeds pf's range.

        With a stride one too small, two different pairs collide and the
        partition silently merges them. Asserted here because the expression is
        easy to copy with the wrong stride.
        """
        _, stripes, cartesian, by_unique, _ = combined_partitions()
        too_small = compress_partition(stripes + int(stripes.max()) * cartesian)
        assert not same_grouping(too_small, by_unique), \
            "the case does not expose a stride that is too small"
        assert too_small.max() < by_unique.max()
        print(f"ok test_the_arithmetic_form_needs_the_right_stride "
              f"(a stride of {int(stripes.max())} loses "
              f"{by_unique.max() - too_small.max()} blocks)")

    test_the_polar_partition_has_sectors_and_a_disc()
    test_the_wrapping_sector_is_disconnected_until_processed()
    test_processing_the_polar_partition_only_splits()
    test_every_polar_block_is_connected_after_processing()
    test_a_partition_needs_no_logical_indices()
    test_the_sign_partition_is_many_disconnected_patches()
    test_the_two_combination_methods_agree()
    test_the_combination_refines_both_inputs()
    test_the_combination_is_finer_than_either_input()
    test_the_arithmetic_form_needs_the_right_stride()
    print("\nAll tests passed!")
