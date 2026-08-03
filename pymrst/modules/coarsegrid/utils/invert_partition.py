"""Block-to-cell lookup from a partition (MRST ``invertPartition.m``).

A partition answers "which block holds this cell". Almost every coarsening
routine needs the other direction, "which cells are in this block", and needs it
for all blocks at once rather than by repeated searching.

The answer is the usual packed pair: ``b2c[b2c_pos[b]:b2c_pos[b + 1]]`` are the
cells of block `b`, the same layout `cells.facePos`/`cells.faces` uses on a grid.
`locno` gives each cell its position inside its own block, which is what turns a
global cell number into an index into a per-block array.

MRST sorts `[p, c]` by row and run-length encodes. A stable argsort of the
partition is the same thing: stable, so cells stay in ascending order inside
each block, which `locno` relies on.
"""

from __future__ import annotations

import numpy as np


def invert_partition(partition, n_blocks: int | None = None):
    """Return ``(b2c_pos, b2c, locno)``.

    `n_blocks` defaults to one past the highest block in use. Pass it
    explicitly when the partition may leave the last blocks empty, since an
    empty trailing block is invisible in the partition itself.
    """
    partition = np.asarray(partition, dtype=np.int64).ravel()
    if partition.size and partition.min() < 0:
        raise ValueError(f"block numbers must be non-negative, got {partition.min()}")
    highest = int(partition.max()) + 1 if partition.size else 0
    if n_blocks is None:
        n_blocks = highest
    elif n_blocks < highest:
        raise ValueError(
            f"n_blocks is {n_blocks} but the partition uses block {highest - 1}")

    b2c = np.argsort(partition, kind="stable").astype(np.int64)
    counts = np.bincount(partition, minlength=n_blocks)
    b2c_pos = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)

    # Position of each cell within its block: its rank in b2c, less the offset
    # of the block it landed in.
    rank = np.empty(partition.size, dtype=np.int64)
    rank[b2c] = np.arange(partition.size, dtype=np.int64)
    locno = rank - b2c_pos[partition]
    return b2c_pos, b2c, locno


__all__ = ["invert_partition"]


if __name__ == "__main__":
    def test_blocks_hold_exactly_their_cells():
        """The defining property, in both directions."""
        p = np.array([2, 0, 1, 2, 0, 0])
        pos, b2c, _ = invert_partition(p)
        for b in range(3):
            cells = b2c[pos[b]:pos[b + 1]]
            assert np.all(p[cells] == b), (b, cells)
            assert cells.size == int(np.count_nonzero(p == b)), b
        print("ok test_blocks_hold_exactly_their_cells")

    def test_every_cell_appears_once():
        p = np.array([2, 0, 1, 2, 0, 0])
        _, b2c, _ = invert_partition(p)
        assert np.array_equal(np.sort(b2c), np.arange(p.size))
        print("ok test_every_cell_appears_once")

    def test_cells_stay_in_ascending_order_inside_a_block():
        """`locno` is only meaningful if the ordering is stable.

        An unstable sort would still list the right cells in each block, and
        `locno` would still be a valid permutation, so nothing would look
        wrong until a per-block array was indexed with it.
        """
        p = np.array([1, 0, 1, 0, 1, 0])
        pos, b2c, _ = invert_partition(p)
        for b in range(2):
            cells = b2c[pos[b]:pos[b + 1]]
            assert np.array_equal(cells, np.sort(cells)), (b, cells)
        print("ok test_cells_stay_in_ascending_order_inside_a_block")

    def test_locno_indexes_into_its_own_block():
        p = np.array([2, 0, 1, 2, 0, 0])
        pos, b2c, locno = invert_partition(p)
        for c in range(p.size):
            assert b2c[pos[p[c]] + locno[c]] == c, c
        print("ok test_locno_indexes_into_its_own_block")

    def test_locno_starts_at_zero_in_each_block():
        p = np.array([2, 0, 1, 2, 0, 0])
        _, _, locno = invert_partition(p)
        for b in range(3):
            inside = locno[p == b]
            assert np.array_equal(np.sort(inside), np.arange(inside.size)), b
        print("ok test_locno_starts_at_zero_in_each_block")

    def test_pos_is_monotone_and_spans_the_cells():
        p = np.array([2, 0, 1, 2, 0, 0])
        pos, _, _ = invert_partition(p)
        assert pos[0] == 0 and pos[-1] == p.size
        assert np.all(np.diff(pos) >= 0)
        print("ok test_pos_is_monotone_and_spans_the_cells")

    def test_an_empty_block_gives_an_empty_range():
        """A gap in the numbering must not shift the blocks after it."""
        p = np.array([0, 0, 2, 2])          # block 1 is empty
        pos, b2c, _ = invert_partition(p)
        assert pos[1] == pos[2], pos
        assert np.array_equal(b2c[pos[2]:pos[3]], np.array([2, 3]))
        print("ok test_an_empty_block_gives_an_empty_range")

    def test_trailing_empty_blocks_need_to_be_declared():
        """They are invisible in the partition, so the caller has to say."""
        p = np.array([0, 0, 1, 1])
        assert invert_partition(p)[0].size == 3
        pos, _, _ = invert_partition(p, n_blocks=4)
        assert pos.size == 5 and pos[2] == pos[4] == 4, pos
        print("ok test_trailing_empty_blocks_need_to_be_declared")

    def test_a_single_block_holds_everything():
        p = np.zeros(5, dtype=np.int64)
        pos, b2c, locno = invert_partition(p)
        assert np.array_equal(pos, [0, 5])
        assert np.array_equal(b2c, np.arange(5))
        assert np.array_equal(locno, np.arange(5))
        print("ok test_a_single_block_holds_everything")

    def test_bad_input_is_rejected():
        try:
            invert_partition(np.array([0, -1]))
        except ValueError as exc:
            assert "non-negative" in str(exc), exc
        else:
            raise AssertionError("expected a negative block to fail")
        try:
            invert_partition(np.array([0, 3]), n_blocks=2)
        except ValueError as exc:
            assert "n_blocks" in str(exc), exc
        else:
            raise AssertionError("expected too few blocks to fail")
        print("ok test_bad_input_is_rejected")

    test_blocks_hold_exactly_their_cells()
    test_every_cell_appears_once()
    test_cells_stay_in_ascending_order_inside_a_block()
    test_locno_indexes_into_its_own_block()
    test_locno_starts_at_zero_in_each_block()
    test_pos_is_monotone_and_spans_the_cells()
    test_an_empty_block_gives_an_empty_range()
    test_trailing_empty_blocks_need_to_be_declared()
    test_a_single_block_holds_everything()
    test_bad_input_is_rejected()
    print("\nAll tests passed!")
