"""Tensor-product partition from explicit block widths (MRST ``partitionTensor.m``).

`partition_ui` divides each logical direction into a given *number* of blocks
and lets the code work out the widths. This takes the widths directly, so the
blocks need not be the same size: `[1, 4, 1]` puts a thin block at each end of a
wide one, which is how a near-boundary or near-fault refinement is expressed.

The widths in each direction must sum to the model's logical extent in that
direction, and that is checked rather than assumed. A short list would silently
leave the far cells in whichever block the arithmetic happened to give them.

Each direction is folded in one at a time, and `process_partition` runs after
each fold, so a block cut in two by inactive cells is split as soon as it
appears rather than at the end. That matters on a grid with a large inactive
region: a block that is disconnected in i and then subdivided in j produces more
pieces if the split is deferred, and MRST's order is the one reproduced here.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from pymrst.modules.coarsegrid.compress_partition import compress_partition
from pymrst.modules.coarsegrid.process_partition import process_partition
from pymrst.utils.gridtools.grid_logical_indices import grid_logical_indices


def partition_tensor(g, *deltas: Sequence[int]) -> np.ndarray:
    """Partition `g` into blocks of the given widths per logical direction.

    Pass one width list per grid dimension: two in 2D, three in 3D. Returns a
    0-based partition.
    """
    griddim = int(getattr(g, "griddim", 3))
    if len(deltas) != griddim:
        raise ValueError(
            f"a {griddim}D grid needs {griddim} block-width lists, "
            f"got {len(deltas)}")

    ijk = np.column_stack([np.asarray(a, dtype=np.int64).ravel()
                           for a in grid_logical_indices(g)[:griddim]])
    extent = ijk.max(axis=0) - ijk.min(axis=0) + 1

    widths = []
    for d, (delta, n) in enumerate(zip(deltas, extent)):
        arr = np.asarray(delta, dtype=np.int64).ravel()
        if arr.size == 0 or np.any(arr < 1):
            raise ValueError(
                f"every block width must be at least 1; direction {d} has "
                f"{arr.tolist()}")
        if int(arr.sum()) != int(n):
            raise ValueError(
                f"block widths in direction {d} sum to {int(arr.sum())} but the "
                f"grid spans {int(n)} logical cells")
        widths.append(arr)

    partition = np.zeros(int(g.cells.num), dtype=np.int64)
    for d in range(griddim - 1, -1, -1):
        arr = widths[d]
        # Block index for each logical index along this direction.
        local = np.repeat(np.arange(arr.size, dtype=np.int64), arr)
        here = local[ijk[:, d] - ijk[:, d].min()]
        stride = int(partition.max()) + 1
        partition = here * stride + partition
        partition = compress_partition(process_partition(g, partition))
    return partition


__all__ = ["partition_tensor"]


if __name__ == "__main__":
    from pymrst.gridprocessing import cart_grid, compute_geometry
    from pymrst.gridprocessing.remove_cells import remove_cells
    from pymrst.modules.coarsegrid.partition_ui import partition_ui

    def _grid(dims=(6, 4, 2)):
        return compute_geometry(cart_grid(list(dims), [float(d) for d in dims]))

    def test_blocks_have_the_requested_widths():
        """The defining property, measured on the grid rather than assumed."""
        g = _grid()
        p = partition_tensor(g, [2, 4], [1, 3], [2])
        ijk = np.column_stack(grid_logical_indices(g))
        assert p.max() + 1 == 4, p.max() + 1
        widths = sorted({tuple(ijk[p == b].max(axis=0) - ijk[p == b].min(axis=0) + 1)
                         for b in range(p.max() + 1)})
        assert widths == sorted([(2, 1, 2), (2, 3, 2), (4, 1, 2), (4, 3, 2)]), widths
        print(f"ok test_blocks_have_the_requested_widths ({p.max() + 1} blocks)")

    def test_unequal_widths_are_the_point():
        """Equal widths are `partition_ui`'s job; this exists for the rest.

        A version that ignored the widths and split evenly would pass every
        count-based check and give the wrong blocks.
        """
        g = _grid((6, 4, 2))
        p = partition_tensor(g, [1, 4, 1], [4], [2])
        sizes = sorted(np.bincount(p).tolist())
        assert sizes == [8, 8, 32], sizes
        print(f"ok test_unequal_widths_are_the_point (block sizes {sizes})")

    def test_it_agrees_with_partition_ui_on_equal_widths():
        """The two are the same partition where they overlap.

        Labels differ, so the grouping is what is compared.
        """
        from pymrst.modules.coarsegrid.utils.is_same_partition import same_grouping
        g = _grid((6, 4, 2))
        a = partition_tensor(g, [3, 3], [2, 2], [2])
        b = partition_ui(g, [2, 2, 1])
        assert same_grouping(a, b), "the two constructors disagree"
        print("ok test_it_agrees_with_partition_ui_on_equal_widths")

    def test_every_cell_is_assigned_exactly_once():
        g = _grid()
        p = partition_tensor(g, [2, 4], [1, 3], [2])
        assert p.size == g.cells.num
        assert np.all(p >= 0)
        assert np.array_equal(np.unique(p), np.arange(p.max() + 1))
        print("ok test_every_cell_is_assigned_exactly_once")

    def test_a_single_block_per_direction_gives_one_block():
        g = _grid()
        p = partition_tensor(g, [6], [4], [2])
        assert p.max() + 1 == 1, p.max() + 1
        print("ok test_a_single_block_per_direction_gives_one_block")

    def test_unit_widths_reproduce_the_fine_grid():
        g = _grid((3, 2, 2))
        p = partition_tensor(g, [1] * 3, [1] * 2, [1] * 2)
        assert p.max() + 1 == g.cells.num, p.max() + 1
        assert len(set(p.tolist())) == g.cells.num
        print("ok test_unit_widths_reproduce_the_fine_grid")

    def test_it_works_in_two_dimensions():
        g = compute_geometry(cart_grid([6, 4], [6.0, 4.0]))
        p = partition_tensor(g, [2, 4], [1, 3])
        assert p.max() + 1 == 4, p.max() + 1
        print("ok test_it_works_in_two_dimensions")

    def test_a_disconnected_block_is_split():
        """Inactive cells can cut a logical block in two.

        Nothing in the tensor construction notices; only `process_partition`
        does, and it runs after each direction is folded in.
        """
        g = compute_geometry(remove_cells(
            cart_grid([5, 1, 1], [5.0, 1.0, 1.0]), np.array([2])))
        p = partition_tensor(g, [5], [1], [1])
        assert p.max() + 1 == 2, p.tolist()
        assert p[0] == p[1] and p[2] == p[3] and p[0] != p[2], p.tolist()
        print("ok test_a_disconnected_block_is_split")

    def test_every_block_is_connected():
        """The postcondition, checked by re-running the splitter."""
        g = compute_geometry(remove_cells(
            cart_grid([6, 4, 1], [6.0, 4.0, 1.0]), np.array([7, 8, 13, 14])))
        p = partition_tensor(g, [3, 3], [2, 2], [1])
        assert np.array_equal(process_partition(g, p), p), "still disconnected"
        print(f"ok test_every_block_is_connected ({p.max() + 1} blocks)")

    def test_widths_that_do_not_span_the_grid_are_rejected():
        """A short list would leave the far cells wherever they landed."""
        g = _grid()
        for args, why in ((([2, 3], [4], [2]), "widths sum too low"),
                          (([2, 4], [5], [2]), "widths sum too high")):
            try:
                partition_tensor(g, *args)
            except ValueError as exc:
                assert "logical cells" in str(exc), exc
            else:
                raise AssertionError(f"expected {why} to be rejected")
        print("ok test_widths_that_do_not_span_the_grid_are_rejected")

    def test_bad_input_is_rejected():
        g = _grid()
        try:
            partition_tensor(g, [6], [4])
        except ValueError as exc:
            assert "block-width lists" in str(exc), exc
        else:
            raise AssertionError("expected too few width lists to fail")
        try:
            partition_tensor(g, [6, 0], [4], [2])
        except ValueError as exc:
            assert "at least 1" in str(exc), exc
        else:
            raise AssertionError("expected a zero width to fail")
        print("ok test_bad_input_is_rejected")

    test_blocks_have_the_requested_widths()
    test_unequal_widths_are_the_point()
    test_it_agrees_with_partition_ui_on_equal_widths()
    test_every_cell_is_assigned_exactly_once()
    test_a_single_block_per_direction_gives_one_block()
    test_unit_widths_reproduce_the_fine_grid()
    test_it_works_in_two_dimensions()
    test_a_disconnected_block_is_split()
    test_every_block_is_connected()
    test_widths_that_do_not_span_the_grid_are_rejected()
    test_bad_input_is_rejected()
    print("\nAll tests passed!")
