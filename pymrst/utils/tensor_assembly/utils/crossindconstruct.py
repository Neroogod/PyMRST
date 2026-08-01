"""Cross-product index pairs (MRST ``tensorAssembly/utils/crossindconstruct.m``).

Given per-group counts `nb1` and `nb2`, produce the index pairs of the
**group-wise cross product**: within group `i`, every one of its `nb1[i]` first
entries paired with every one of its `nb2[i]` second entries. Both outputs
index the concatenated arrays, so group offsets are already folded in.

    nb1 = [2, 1], nb2 = [3, 2]
    -> 2*3 + 1*2 = 8 pairs
       crossind1 = 0 0 0 1 1 1 2 2
       crossind2 = 0 1 2 0 1 2 3 4

This is the primitive underneath `crossIndexArray`: joining two index arrays on
a shared index means taking that cross product within each matching group.

MRST builds it from four interleaved `cumsum` passes over difference arrays,
and that construction has **two failure modes this one does not**:

- a **trailing** group with `nb2 = 0` leaves the arrays mismatched and it
  raises "Arrays have incompatible sizes";
- an **interior** group with `nb1 = 0` makes two write positions collide, so
  the running offset loses that group's slots and later groups point into the
  wrong block. With `nb1 = [2,0,1]`, `nb2 = [3,5,2]`, group 2's second entries
  belong at slots `[8, 9]`; MRST answers `[5, 6]`, inside group 1's block.

The same result is two `repeat`s and an `mcolon`: repeat each first entry as
many times as its group has second entries, and lay the group's second entries
out beneath. Both are verified against MRST on the cases it gets right - see
`scripts/validate_crossindconstruct.py`.
"""

import numpy as np

from pymrst.utils.mcolon import mcolon


def crossindconstruct(nb1, nb2):
    """Return ``(crossind1, crossind2)``, 0-based into the concatenated arrays."""
    nb1 = np.asarray(nb1, dtype=np.int64).reshape(-1)
    nb2 = np.asarray(nb2, dtype=np.int64).reshape(-1)
    if nb1.size != nb2.size:
        raise ValueError(
            f"one count per group is needed for each side, got {nb1.size} "
            f"and {nb2.size}")
    if np.any(nb1 < 0) or np.any(nb2 < 0):
        raise ValueError("counts cannot be negative")

    # Each first entry pairs with every second entry of its own group.
    crossind1 = np.repeat(np.arange(nb1.sum(), dtype=np.int64),
                          np.repeat(nb2, nb1))

    # ... and the group's whole second block is laid out under each of them.
    starts2 = np.concatenate([[0], np.cumsum(nb2)[:-1]]) if nb2.size else nb2
    crossind2 = mcolon(np.repeat(starts2, nb1), np.repeat(starts2 + nb2, nb1))
    return crossind1, crossind2.astype(np.int64)


if __name__ == "__main__":

    def test_the_worked_example() -> None:
        a, b = crossindconstruct([2, 1], [3, 2])
        assert a.tolist() == [0, 0, 0, 1, 1, 1, 2, 2], a.tolist()
        assert b.tolist() == [0, 1, 2, 0, 1, 2, 3, 4], b.tolist()
        print("ok test_the_worked_example")

    def test_the_pair_count_is_the_sum_of_products() -> None:
        for nb1, nb2 in (([2, 1], [3, 2]), ([1], [1]), ([4, 0, 2], [2, 5, 3])):
            a, b = crossindconstruct(nb1, nb2)
            expected = int(np.dot(nb1, nb2))
            assert a.size == b.size == expected, (nb1, nb2, a.size, expected)
        print("ok test_the_pair_count_is_the_sum_of_products")

    def test_it_is_the_cross_product_within_each_group() -> None:
        """Checked against the obvious nested-loop construction."""
        nb1, nb2 = [3, 2, 1], [2, 4, 3]
        a, b = crossindconstruct(nb1, nb2)
        want1, want2, o1, o2 = [], [], 0, 0
        for n1, n2 in zip(nb1, nb2):
            for i in range(n1):
                for j in range(n2):
                    want1.append(o1 + i)
                    want2.append(o2 + j)
            o1 += n1
            o2 += n2
        assert a.tolist() == want1 and b.tolist() == want2
        print("ok test_it_is_the_cross_product_within_each_group")

    def test_an_empty_group_contributes_nothing() -> None:
        """An empty group produces no pairs but still occupies its slots.

        This is where MRST goes wrong: it loses group 1's five slots and points
        group 2 at [5, 6] instead of [8, 9]. See the module docstring.
        """
        a, b = crossindconstruct([2, 0, 1], [3, 5, 2])
        assert a.size == 2 * 3 + 0 * 5 + 1 * 2
        assert b.tolist()[-2:] == [8, 9], b.tolist()
        assert b.max() == 3 + 5 + 2 - 1
        print("ok test_an_empty_group_contributes_nothing")

    def test_a_zero_on_the_second_side_drops_its_group() -> None:
        a, b = crossindconstruct([2, 3], [4, 0])
        assert a.size == 8 and set(a.tolist()) == {0, 1}
        print("ok test_a_zero_on_the_second_side_drops_its_group")

    def test_no_groups_gives_nothing() -> None:
        a, b = crossindconstruct([], [])
        assert a.size == 0 and b.size == 0
        print("ok test_no_groups_gives_nothing")

    def test_mismatched_group_counts_are_rejected() -> None:
        try:
            crossindconstruct([1, 2], [1])
        except ValueError as error:
            assert "one count per group" in str(error)
            print("ok test_mismatched_group_counts_are_rejected")
            return
        raise AssertionError("mismatched group counts must be rejected")

    def test_negative_counts_are_rejected() -> None:
        try:
            crossindconstruct([1, -2], [1, 1])
        except ValueError as error:
            assert "cannot be negative" in str(error)
            print("ok test_negative_counts_are_rejected")
            return
        raise AssertionError("negative counts must be rejected")

    test_the_worked_example()
    test_the_pair_count_is_the_sum_of_products()
    test_it_is_the_cross_product_within_each_group()
    test_an_empty_group_contributes_nothing()
    test_a_zero_on_the_second_side_drops_its_group()
    test_no_groups_gives_nothing()
    test_mismatched_group_counts_are_rejected()
    test_negative_counts_are_rejected()
    print("\nAll tests passed!")
