"""Renumber a partition to remove empty blocks (MRST ``compressPartition.m``).

A partition assigns every fine cell to a coarse block. Operations that split,
merge or filter blocks leave gaps: block 3 may end up with no cells while blocks
4 and 5 still exist. `compress_partition` renumbers so the block ids are
contiguous again, preserving which cells share a block.

Every partition constructor in this module ends by calling it, which is why it
comes first.

**Indexing.** MRST's partitions are 1-based and its version shifts any
non-positive minimum up to 1. PyMRST's are 0-based, matching
`partition_cart_grid`, and the port shifts a negative minimum up to 0 instead.
The two conventions produce the same grouping; only the labels differ.
"""

from __future__ import annotations

import numpy as np


def compress_partition(p) -> np.ndarray:
    """Renumber `p` so its block ids are contiguous from 0.

    Cells that shared a block still share one, and the relative order of the
    surviving block ids is preserved: the lowest original id becomes 0, the next
    becomes 1, and so on. Returns a new array; the input is untouched.
    """
    p = np.asarray(p, dtype=np.int64).ravel()
    if p.size == 0:
        return p.copy()

    # `np.unique` gives the surviving ids in ascending order and, with
    # `return_inverse`, the position of each cell's id within that list. That
    # position *is* the compressed label, so the whole operation is one call.
    # MRST builds the same map through `accumarray` and an explicit lookup.
    _, inverse = np.unique(p, return_inverse=True)
    return inverse.astype(np.int64).reshape(-1)


__all__ = ["compress_partition"]


if __name__ == "__main__":
    def test_gaps_are_removed():
        """Blocks 0, 2 and 5 become 0, 1 and 2."""
        got = compress_partition([0, 0, 2, 5, 5, 2])
        assert got.tolist() == [0, 0, 1, 2, 2, 1], got.tolist()
        print("ok test_gaps_are_removed")

    def test_membership_is_preserved():
        """The whole point: which cells share a block must not change.

        Renumbering that reassigns cells is not compression, and the result
        still looks like a valid partition, so the grouping is asserted rather
        than the labels alone.
        """
        p = np.array([7, 3, 7, 9, 3, 9, 9])
        q = compress_partition(p)
        for i in range(p.size):
            for j in range(p.size):
                assert (p[i] == p[j]) == (q[i] == q[j]), (i, j)
        print("ok test_membership_is_preserved")

    def test_already_contiguous_is_unchanged():
        """MRST notes that a gapless partition comes back as it went in."""
        p = np.array([0, 1, 2, 1, 0])
        assert compress_partition(p).tolist() == p.tolist()
        print("ok test_already_contiguous_is_unchanged")

    def test_result_is_contiguous_from_zero():
        for case in ([4, 4, 9], [100], [3, 1, 2], list(range(5, 15))):
            q = compress_partition(case)
            assert q.min() == 0, (case, q.min())
            assert np.array_equal(np.unique(q), np.arange(q.max() + 1)), q
        print("ok test_result_is_contiguous_from_zero")

    def test_order_of_surviving_ids_is_preserved():
        """The lowest original id maps to 0, the next to 1, and so on."""
        q = compress_partition([50, 10, 30, 10])
        assert q.tolist() == [2, 0, 1, 0], q.tolist()
        print("ok test_order_of_surviving_ids_is_preserved")

    def test_negative_labels_are_shifted_up():
        """A negative minimum is legal input and comes back based at 0."""
        q = compress_partition([-3, -3, 0, 4])
        assert q.tolist() == [0, 0, 1, 2], q.tolist()
        assert q.min() == 0
        print("ok test_negative_labels_are_shifted_up")

    def test_input_is_not_modified():
        p = np.array([5, 5, 9])
        before = p.copy()
        out = compress_partition(p)
        assert np.array_equal(p, before), "input was mutated"
        out[0] = 99
        assert p[0] == 5, "result aliases the input"
        print("ok test_input_is_not_modified")

    def test_shapes_are_flattened():
        q = compress_partition(np.array([[2, 2], [7, 7]]))
        assert q.shape == (4,), q.shape
        assert q.tolist() == [0, 0, 1, 1]
        assert compress_partition([]).size == 0
        print("ok test_shapes_are_flattened")

    def test_idempotent():
        """Compressing twice changes nothing the second time."""
        p = [9, 1, 1, 4]
        once = compress_partition(p)
        assert np.array_equal(compress_partition(once), once)
        print("ok test_idempotent")

    test_gaps_are_removed()
    test_membership_is_preserved()
    test_already_contiguous_is_unchanged()
    test_result_is_contiguous_from_zero()
    test_order_of_surviving_ids_is_preserved()
    test_negative_labels_are_shifted_up()
    test_input_is_not_modified()
    test_shapes_are_flattened()
    test_idempotent()
    print("\nAll tests passed!")
