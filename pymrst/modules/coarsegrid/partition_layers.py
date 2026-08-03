"""Uniform in I and J, layered in K (MRST ``partitionLayers.m``).

Partitions a logically Cartesian grid uniformly in the two areal directions and
by explicitly named layer groups vertically. That combination is what
geological models usually want: areal blocks of roughly equal size, but vertical
blocks that follow stratigraphy rather than arithmetic.

Layers are given as boundaries, not counts. `layers = [1, 4, 8, 12]` means three
vertical groups holding K in 1 to 3, 4 to 7 and 8 to 11. MRST expands them with
`rldecode(1:numel(L)-1, diff(L), 2)`, and the port does the same with
`np.repeat`.

The areal half is `lb_lin_dist` from `partition_ui`, so a fine dimension that
does not divide evenly spreads its remainder over the leading blocks. MRST keeps
a private copy of that function in each file; this imports the single one.

**Indexing.** Layer boundaries are 1-based, as MRST writes them, because they
name stratigraphic layers and renumbering them would invite the wrong groups.
The returned block index is 0-based, like the rest of this module.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pymrst.modules.coarsegrid.partition_ui import lb_lin_dist


def partition_layers(g, coarse_dim: Sequence[int], layers: Sequence[int]) -> np.ndarray:
    """Partition `g` into `coarse_dim` areal blocks and the given layer groups.

    `coarse_dim` holds two entries, for I and J. `layers` holds the 1-based K
    boundaries of each vertical group, ascending, with the final entry one past
    the last layer included.

    Returns a 0-based block index per cell.
    """
    if getattr(g, "cartDims", None) is None:
        raise ValueError(
            "partition_layers needs a logically Cartesian grid; this one has "
            "no cartDims")
    index_map = getattr(g.cells, "indexMap", None)
    if index_map is None:
        raise ValueError("partition_layers needs cells.indexMap")

    coarse = np.asarray(coarse_dim, dtype=np.int64).ravel()
    if coarse.size != 2:
        raise ValueError(
            f"coarse_dim names the areal (I, J) blocks and needs two entries, "
            f"got {coarse.size}")
    if np.any(coarse < 1):
        raise ValueError(f"coarse_dim must be strictly positive, got {coarse.tolist()}")

    bounds = np.asarray(layers, dtype=np.int64).ravel()
    if bounds.size < 2:
        raise ValueError(
            "layers gives the boundaries of each vertical group and needs at "
            "least two entries, for example [1, nz + 1]")
    if np.any(np.diff(bounds) <= 0):
        raise ValueError(f"layer boundaries must increase, got {bounds.tolist()}")

    dims = np.asarray(g.cartDims, dtype=np.int64).ravel()
    nz = int(dims[2]) if dims.size > 2 else 1
    if bounds[0] < 1 or bounds[-1] > nz + 1:
        raise ValueError(
            f"layer boundaries {bounds.tolist()} lie outside 1 to {nz + 1}; "
            f"they are 1-based and the last entry is one past the final layer")

    ijk = np.array(np.unravel_index(
        np.asarray(index_map, dtype=np.int64), tuple(dims), order="F")).T

    # Group number per K index, 0-based. Layers below the first boundary or at
    # or above the last are not covered by any group; MRST indexes into a short
    # array and would fail, so the port says which cells are unassigned.
    group_of_k = np.full(nz, -1, dtype=np.int64)
    counts = np.diff(bounds)
    covered = np.repeat(np.arange(counts.size, dtype=np.int64), counts)
    group_of_k[bounds[0] - 1:bounds[-1] - 1] = covered

    k_index = ijk[:, 2] if ijk.shape[1] > 2 else np.zeros(ijk.shape[0], dtype=np.int64)
    block = group_of_k[k_index]
    if np.any(block < 0):
        missing = np.unique(k_index[block < 0]) + 1
        raise ValueError(
            f"layers {bounds.tolist()} do not cover K = {missing.tolist()}")

    span = ijk.max(axis=0) - ijk.min(axis=0) + 1
    for d in (1, 0):
        b = int(coarse[d])
        block = lb_lin_dist(ijk[:, d] - ijk[:, d].min(), int(span[d]), b) + b * block
    return block


__all__ = ["partition_layers"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.modules.coarsegrid.compress_partition import compress_partition

    def _grid(dims=(6, 4, 6)):
        return compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))

    def test_layers_are_boundaries_not_counts():
        """`[1, 3, 7]` means two groups: K in 1-2 and K in 3-6.

        Reading them as counts instead gives a partition of the right shape
        holding entirely the wrong layers, which no downstream check notices.
        """
        g = _grid((4, 4, 6))
        p = partition_layers(g, [1, 1], [1, 3, 7])
        dims = tuple(np.asarray(g.cartDims, dtype=np.int64))
        _, _, k = np.unravel_index(
            np.asarray(g.cells.indexMap, dtype=np.int64), dims, order="F")
        assert p.max() + 1 == 2, p.max() + 1
        assert set(k[p == 0].tolist()) == {0, 1}, sorted(set(k[p == 0].tolist()))
        assert set(k[p == 1].tolist()) == {2, 3, 4, 5}, sorted(set(k[p == 1].tolist()))
        print("ok test_layers_are_boundaries_not_counts")

    def test_block_count_is_areal_times_layers():
        g = _grid((6, 4, 6))
        p = partition_layers(g, [3, 2], [1, 3, 5, 7])
        assert p.max() + 1 == 3 * 2 * 3, p.max() + 1
        print(f"ok test_block_count_is_areal_times_layers ({p.max() + 1} blocks)")

    def test_partition_covers_every_cell_once():
        g = _grid((6, 4, 6))
        p = partition_layers(g, [3, 2], [1, 4, 7])
        assert p.shape == (g.cells.num,)
        assert np.array_equal(np.unique(p), np.arange(p.max() + 1))
        print("ok test_partition_covers_every_cell_once")

    def test_each_block_lies_in_one_layer_group():
        """A block must not straddle the layer boundaries it was built from."""
        g = _grid((6, 4, 6))
        bounds = [1, 3, 7]
        p = partition_layers(g, [3, 2], bounds)
        dims = tuple(np.asarray(g.cartDims, dtype=np.int64))
        _, _, k = np.unravel_index(
            np.asarray(g.cells.indexMap, dtype=np.int64), dims, order="F")
        edges = np.asarray(bounds) - 1
        for b in range(p.max() + 1):
            kk = k[p == b]
            group = np.searchsorted(edges, kk.min(), side="right") - 1
            assert np.all((kk >= edges[group]) & (kk < edges[group + 1])), (b, kk)
        print("ok test_each_block_lies_in_one_layer_group")

    def test_uneven_layers_are_allowed():
        """Groups need not be the same thickness; that is the point."""
        g = _grid((4, 4, 6))
        p = partition_layers(g, [1, 1], [1, 2, 6, 7])
        sizes = np.bincount(p)
        assert sizes.tolist() == [16, 64, 16], sizes.tolist()
        print(f"ok test_uneven_layers_are_allowed (block sizes {sizes.tolist()})")

    def test_areal_remainder_is_balanced():
        """The guarantee is per direction, not on the block sizes themselves.

        `lb_lin_dist` makes the cell counts along I differ by at most one, and
        likewise along J. Block *volumes* are products of those counts, so they
        can differ by much more: a 7x5x4 grid partitioned 3x2 gives
        [36, 24, 24, 24, 16, 16], which is correct. Asserting a bound on the
        volumes would either be arbitrary or wrong, so the per-direction
        property is checked directly.
        """
        g = _grid((7, 5, 4))
        p = partition_layers(g, [3, 2], [1, 5])
        dims = tuple(np.asarray(g.cartDims, dtype=np.int64))
        i, j, _ = np.unravel_index(
            np.asarray(g.cells.indexMap, dtype=np.int64), dims, order="F")
        i_widths = sorted({len(np.unique(i[p == b])) for b in range(p.max() + 1)})
        j_widths = sorted({len(np.unique(j[p == b])) for b in range(p.max() + 1)})
        assert max(i_widths) - min(i_widths) <= 1, i_widths
        assert max(j_widths) - min(j_widths) <= 1, j_widths
        assert np.bincount(p).sum() == g.cells.num
        print(f"ok test_areal_remainder_is_balanced "
              f"(I widths {i_widths}, J widths {j_widths})")

    def test_result_needs_no_compression():
        g = _grid((7, 5, 6))
        p = partition_layers(g, [3, 2], [1, 3, 7])
        assert np.array_equal(compress_partition(p), p), "gaps in the partition"
        print("ok test_result_needs_no_compression")

    def test_uncovered_layers_are_reported():
        """Layers outside the boundaries are named rather than silently dropped.

        MRST indexes into a short lookup here and fails obscurely; saying which
        K values are uncovered is the difference between a fixable message and
        a puzzle.
        """
        g = _grid((4, 4, 6))
        try:
            partition_layers(g, [1, 1], [1, 4])       # covers K 1-3, not 4-6
        except ValueError as exc:
            assert "do not cover K" in str(exc), exc
            assert "4" in str(exc), exc
        else:
            raise AssertionError("expected uncovered layers to be reported")
        print("ok test_uncovered_layers_are_reported")

    def test_invalid_arguments_are_rejected():
        g = _grid((4, 4, 4))
        for args, why in ((([1, 1, 1], [1, 5]), "three areal dims"),
                          (([0, 1], [1, 5]), "non-positive"),
                          (([1, 1], [1]), "one boundary"),
                          (([1, 1], [3, 1]), "descending"),
                          (([1, 1], [1, 99]), "past the last layer")):
            try:
                partition_layers(g, *args)
            except ValueError:
                pass
            else:
                raise AssertionError(f"expected {why} to be rejected: {args}")
        print("ok test_invalid_arguments_are_rejected")

    test_layers_are_boundaries_not_counts()
    test_block_count_is_areal_times_layers()
    test_partition_covers_every_cell_once()
    test_each_block_lies_in_one_layer_group()
    test_uneven_layers_are_allowed()
    test_areal_remainder_is_balanced()
    test_result_needs_no_compression()
    test_uncovered_layers_are_reported()
    test_invalid_arguments_are_rejected()
    print("\nAll tests passed!")
