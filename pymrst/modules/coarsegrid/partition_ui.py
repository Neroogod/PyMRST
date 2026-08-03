"""Uniform partition in logical space (MRST ``partitionUI.m``).

Divides a logically Cartesian grid into `coarse_dim` blocks per direction, as
evenly as the fine dimensions allow. When a direction does not divide exactly,
the remainder is spread over the leading blocks rather than dumped on the last
one, which is what `lb_lin_dist` below does.

`lb_lin_dist` is shared with `partition_layers`. MRST carries a private copy in
each file; this module defines it once and imports it there, so a fix reaches
both.

**Indexing.** MRST returns 1-based block numbers. PyMRST returns 0-based, in
line with `partition_cart_grid` and `compress_partition`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def lb_lin_dist(f: np.ndarray, m: int, b: int) -> np.ndarray:
    """Load-balanced distribution of `m` positions into `b` blocks.

        L = m // b      cells per block
        R = m %  b      cells left over
        out = max(f // (L + 1), (f - R) // L)

    The first `R` blocks take `L + 1` cells and the rest take `L`, so the
    largest and smallest blocks differ by at most one. Distributing the
    remainder onto a single block instead would still produce a valid
    partition, with one block several times the size of its neighbours.
    """
    f = np.asarray(f, dtype=np.int64)
    m, b = int(m), int(b)
    if b < 1:
        raise ValueError(f"block count must be positive, got {b}")
    per, rem = m // b, m % b
    if per == 0:
        # More blocks than cells: each cell is its own block.
        return np.minimum(f, b - 1)
    return np.maximum(f // (per + 1), (f - rem) // per)


def partition_ui(g, coarse_dim: Sequence[int]) -> np.ndarray:
    """Partition `g` uniformly into `coarse_dim` blocks per logical direction.

    Returns a 0-based block index per cell.
    """
    if getattr(g, "cartDims", None) is None:
        raise ValueError(
            "partition_ui needs a logically Cartesian grid; this one has no "
            "cartDims")
    index_map = getattr(g.cells, "indexMap", None)
    if index_map is None:
        raise ValueError("partition_ui needs cells.indexMap")

    dims = np.asarray(g.cartDims, dtype=np.int64).ravel()
    coarse = np.asarray(coarse_dim, dtype=np.int64).ravel()
    if coarse.size != int(g.griddim):
        raise ValueError(
            f"coarse_dim needs {int(g.griddim)} entries for a "
            f"{int(g.griddim)}-D grid, got {coarse.size}")
    if np.any(coarse < 1):
        raise ValueError(f"coarse_dim must be strictly positive, got {coarse.tolist()}")
    if np.any(coarse > dims[:coarse.size]):
        raise ValueError(
            f"coarse_dim {coarse.tolist()} exceeds cartDims "
            f"{dims[:coarse.size].tolist()}")

    ijk = np.array(np.unravel_index(
        np.asarray(index_map, dtype=np.int64), tuple(dims), order="F")).T
    # The extent actually occupied, which is not the full box when inactive
    # cells have been removed. MRST measures the same span rather than using
    # cartDims directly.
    span = ijk.max(axis=0) - ijk.min(axis=0) + 1

    block = np.zeros(int(g.cells.num), dtype=np.int64)
    for d in range(coarse.size - 1, -1, -1):
        b = int(coarse[d])
        block = lb_lin_dist(ijk[:, d] - ijk[:, d].min(), int(span[d]), b) + b * block
    return block


__all__ = ["lb_lin_dist", "partition_ui"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.compress_partition import compress_partition

    def test_even_division():
        """6 cells into 3 blocks gives 2 each."""
        got = lb_lin_dist(np.arange(6), 6, 3)
        assert got.tolist() == [0, 0, 1, 1, 2, 2], got.tolist()
        print("ok test_even_division")

    def test_remainder_goes_to_the_leading_blocks():
        """7 into 3 gives 3, 2, 2, not 2, 2, 3.

        Spreading the remainder is the whole purpose of the formula: dumping it
        on one block leaves a partition that is valid but badly balanced, and
        nothing downstream would complain.
        """
        got = lb_lin_dist(np.arange(7), 7, 3)
        sizes = np.bincount(got)
        assert sizes.tolist() == [3, 2, 2], sizes.tolist()
        assert sizes.max() - sizes.min() <= 1
        print("ok test_remainder_goes_to_the_leading_blocks")

    def test_block_sizes_never_differ_by_more_than_one():
        for m in range(1, 40):
            for b in range(1, min(m, 12) + 1):
                sizes = np.bincount(lb_lin_dist(np.arange(m), m, b), minlength=b)
                assert sizes.sum() == m, (m, b)
                assert sizes.max() - sizes.min() <= 1, (m, b, sizes.tolist())
        print("ok test_block_sizes_never_differ_by_more_than_one")

    def test_partition_covers_every_cell_once():
        """The defining invariant of a partition."""
        g = compute_geometry(cart_grid([6, 4, 2], [6.0, 4.0, 2.0]))
        p = partition_ui(g, [3, 2, 1])
        assert p.shape == (g.cells.num,), p.shape
        assert p.min() == 0
        assert np.array_equal(np.unique(p), np.arange(p.max() + 1))
        assert p.max() + 1 == 3 * 2 * 1, p.max() + 1
        print(f"ok test_partition_covers_every_cell_once ({p.max() + 1} blocks)")

    def test_block_count_matches_the_request():
        g = compute_geometry(cart_grid([8, 6, 4], [8.0, 6.0, 4.0]))
        for dims in ([2, 2, 2], [4, 3, 1], [1, 1, 4], [8, 6, 4]):
            p = partition_ui(g, dims)
            assert p.max() + 1 == int(np.prod(dims)), (dims, p.max() + 1)
        print("ok test_block_count_matches_the_request")

    def test_one_block_per_cell_at_full_resolution():
        g = compute_geometry(cart_grid([3, 2, 1], [3.0, 2.0, 1.0]))
        p = partition_ui(g, [3, 2, 1])
        assert p.max() + 1 == g.cells.num
        assert len(set(p.tolist())) == g.cells.num, "cells share a block"
        print("ok test_one_block_per_cell_at_full_resolution")

    def test_a_single_block_holds_everything():
        g = compute_geometry(cart_grid([5, 4, 3], [5.0, 4.0, 3.0]))
        p = partition_ui(g, [1, 1, 1])
        assert p.max() == 0 and p.size == g.cells.num
        print("ok test_a_single_block_holds_everything")

    def test_blocks_are_logically_contiguous():
        """Each block should be a box in index space, not a scattering.

        A partition that assigns the right *number* of cells to each block
        while interleaving them is a classic indexing error and produces a
        coarse grid that still builds.
        """
        g = compute_geometry(cart_grid([6, 4, 1], [6.0, 4.0, 1.0]))
        p = partition_ui(g, [3, 2, 1])
        dims = tuple(np.asarray(g.cartDims, dtype=np.int64))
        i, j, k = np.unravel_index(
            np.asarray(g.cells.indexMap, dtype=np.int64), dims, order="F")
        for b in range(p.max() + 1):
            sel = p == b
            for axis in (i, j):
                v = axis[sel]
                assert v.max() - v.min() + 1 == np.unique(v).size, (b, v)
        print("ok test_blocks_are_logically_contiguous")

    def test_result_needs_no_compression():
        g = compute_geometry(cart_grid([7, 5, 3], [7.0, 5.0, 3.0]))
        p = partition_ui(g, [3, 2, 2])
        assert np.array_equal(compress_partition(p), p), "gaps in the partition"
        print("ok test_result_needs_no_compression")

    def test_invalid_requests_are_rejected():
        g = compute_geometry(cart_grid([4, 4, 2], [4.0, 4.0, 2.0]))
        for bad, why in (([2, 2], "wrong length"), ([0, 2, 1], "non-positive"),
                         ([9, 2, 1], "exceeds cartDims")):
            try:
                partition_ui(g, bad)
            except ValueError:
                pass
            else:
                raise AssertionError(f"expected {why} to be rejected: {bad}")
        print("ok test_invalid_requests_are_rejected")

    test_even_division()
    test_remainder_goes_to_the_leading_blocks()
    test_block_sizes_never_differ_by_more_than_one()
    test_partition_covers_every_cell_once()
    test_block_count_matches_the_request()
    test_one_block_per_cell_at_full_resolution()
    test_a_single_block_holds_everything()
    test_blocks_are_logically_contiguous()
    test_result_needs_no_compression()
    test_invalid_requests_are_rejected()
    print("\nAll tests passed!")
