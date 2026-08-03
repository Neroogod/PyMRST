"""Do two partitions describe the same blocks (MRST ``isSamePartition.m``)?

Block numbers are labels. Two partitions that group the cells identically but
number the groups differently are the same partition, and code that compares
them elementwise will disagree. That happens routinely: `compress_partition`
renumbers, `process_partition` appends split components at the end, and a
partition round-tripped through a coarse grid comes back relabelled.

The test is that the relation is a *function* in one direction: every block of
`p1` must lie inside a single block of `p2`. MRST checks that by counting the
distinct `(p1, p2)` pairs against the distinct values of `p1`. If some block of
`p1` straddles two blocks of `p2` it contributes two pairs, and the counts
differ.

**This is one-directional, and MRST's name understates it.** A refinement
passes: if `p2` splits a block of `p1`, the pair count goes up and the answer is
`False`, but if `p1` splits a block of `p2` the answer is `True`. So
``is_same_partition(fine, coarse)`` and ``is_same_partition(coarse, fine)`` can
disagree. The port keeps MRST's behaviour and states it, and offers
`is_refinement_of` for the case where the direction is what you actually want.
"""

from __future__ import annotations

import numpy as np


def is_same_partition(p1, p2) -> bool:
    """True when every block of `p1` lies inside one block of `p2`.

    Equal arrays short-circuit, as in MRST. Arrays of different length are
    never the same partition.
    """
    a = np.asarray(p1)
    b = np.asarray(p2)
    if a.shape == b.shape and np.array_equal(a, b):
        return True
    a = a.ravel()
    b = b.ravel()
    if a.size != b.size:
        return False
    pairs = np.unique(np.column_stack([a, b]), axis=0)
    return pairs.shape[0] == np.unique(a).size


def is_refinement_of(fine, coarse) -> bool:
    """True when `fine` splits `coarse` without crossing any of its blocks.

    Not in MRST. `is_same_partition` answers this in one direction only, and
    which direction is easy to get backwards.
    """
    return is_same_partition(fine, coarse)


def same_grouping(p1, p2) -> bool:
    """True when the two partitions group the cells identically.

    The symmetric relation, which is what "same partition" usually means in
    conversation. Not in MRST.
    """
    return is_same_partition(p1, p2) and is_same_partition(p2, p1)


__all__ = ["is_same_partition", "is_refinement_of", "same_grouping"]


if __name__ == "__main__":
    def test_identical_partitions_match():
        p = np.array([0, 0, 1, 1, 2])
        assert is_same_partition(p, p.copy())
        print("ok test_identical_partitions_match")

    def test_relabelling_does_not_change_the_partition():
        """The reason the function exists.

        `compress_partition` and `process_partition` both renumber, so an
        elementwise comparison reports a difference where there is none.
        """
        a = np.array([0, 0, 1, 1, 2])
        b = np.array([2, 2, 0, 0, 1])
        assert not np.array_equal(a, b)
        assert is_same_partition(a, b)
        print("ok test_relabelling_does_not_change_the_partition")

    def test_a_genuinely_different_grouping_is_rejected():
        a = np.array([0, 0, 1, 1])
        b = np.array([0, 1, 0, 1])
        assert not is_same_partition(a, b)
        print("ok test_a_genuinely_different_grouping_is_rejected")

    def test_a_merge_is_rejected_and_a_split_is_not():
        """MRST's test is one-directional, which is easy to trip over.

        `p1` finer than `p2` passes, `p2` finer than `p1` fails. Asserting
        both directions here keeps the asymmetry visible instead of leaving a
        caller to discover it.
        """
        coarse = np.array([0, 0, 0, 0])
        fine = np.array([0, 0, 1, 1])
        assert is_same_partition(fine, coarse), "a refinement should pass"
        assert not is_same_partition(coarse, fine), "a coarsening should fail"
        print("ok test_a_merge_is_rejected_and_a_split_is_not")

    def test_same_grouping_is_symmetric():
        coarse = np.array([0, 0, 0, 0])
        fine = np.array([0, 0, 1, 1])
        assert not same_grouping(coarse, fine)
        assert not same_grouping(fine, coarse)
        assert same_grouping(fine, np.array([1, 1, 0, 0]))
        print("ok test_same_grouping_is_symmetric")

    def test_is_refinement_of_reads_the_right_way_round():
        coarse = np.array([0, 0, 1, 1])
        fine = np.array([0, 1, 2, 3])
        assert is_refinement_of(fine, coarse)
        assert not is_refinement_of(coarse, fine)
        print("ok test_is_refinement_of_reads_the_right_way_round")

    def test_shape_does_not_matter():
        a = np.array([0, 0, 1])
        assert is_same_partition(a, a.reshape(-1, 1))
        print("ok test_shape_does_not_matter")

    def test_different_lengths_are_never_the_same():
        assert not is_same_partition(np.array([0, 0, 1]), np.array([0, 0]))
        print("ok test_different_lengths_are_never_the_same")

    def test_it_survives_compress_partition():
        from pymrst.modules.coarsegrid.compress_partition import compress_partition
        p = np.array([0, 0, 5, 5, 2])
        assert is_same_partition(p, compress_partition(p))
        print("ok test_it_survives_compress_partition")

    def test_it_survives_process_partition():
        """A split partition is a refinement, so it passes one way only."""
        from pymrst.gridprocessing import cart_grid, compute_geometry
        from pymrst.modules.coarsegrid.process_partition import process_partition
        g = compute_geometry(cart_grid([4, 1, 1], [4.0, 1.0, 1.0]))
        p = np.array([0, 1, 1, 0])
        q = process_partition(g, p)
        assert is_same_partition(q, p), "the split should refine the original"
        assert not is_same_partition(p, q), "and not the other way round"
        print("ok test_it_survives_process_partition")

    test_identical_partitions_match()
    test_relabelling_does_not_change_the_partition()
    test_a_genuinely_different_grouping_is_rejected()
    test_a_merge_is_rejected_and_a_split_is_not()
    test_same_grouping_is_symmetric()
    test_is_refinement_of_reads_the_right_way_round()
    test_shape_does_not_matter()
    test_different_lengths_are_never_the_same()
    test_it_survives_compress_partition()
    test_it_survives_process_partition()
    print("\nAll tests passed!")
