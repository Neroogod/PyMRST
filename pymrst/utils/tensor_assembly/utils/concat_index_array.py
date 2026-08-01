"""Stack two index arrays (MRST ``tensorAssembly/utils/concatIndexArray.m``).

Concatenates the rows of `tbl1` and `tbl2` over a chosen set of index names,
giving the union of the two subspaces.

**By default it refuses to produce a duplicate row**, because for an index
array a repeated multi-index is almost always a mistake - the same (cell, face)
pair listed twice will be summed twice by anything that assembles over it.
`remove_duplicates=True` asks for the set union instead.

MRST's guard reads `if (opt.checkUnique) & ~opt.removeDuplicates`, using the
elementwise `&` where `&&` is meant. It happens to work on scalars, so the
behaviour is right; the port states the intent directly.
"""

from typing import Sequence

import numpy as np

from pymrst.utils.tensor_assembly.index_array import IndexArray


def concat_index_array(tbl1, tbl2, fdnames: Sequence[str], *,
                       check_unique: bool = True,
                       remove_duplicates: bool = False) -> IndexArray:
    """Return the rows of both arrays over ``fdnames``, stacked."""
    fdnames = list(fdnames)
    inds = np.vstack([tbl1.gets(fdnames), tbl2.gets(fdnames)])

    if remove_duplicates:
        inds = np.unique(inds, axis=0)
    elif check_unique:
        unique = np.unique(inds, axis=0)
        if unique.shape[0] != inds.shape[0]:
            repeated = inds.shape[0] - unique.shape[0]
            raise ValueError(
                f"{repeated} multi-index would appear more than once; anything "
                f"assembling over this array would count them twice. Pass "
                f"remove_duplicates=True for the set union.")
    return IndexArray(fdnames=fdnames, inds=inds)


concatIndexArray = concat_index_array


if __name__ == "__main__":

    def a():
        return IndexArray({"cells": [0, 1], "faces": [3, 4]})

    def b():
        return IndexArray({"cells": [2, 3], "faces": [5, 6]})

    def test_it_stacks_disjoint_arrays() -> None:
        got = concat_index_array(a(), b(), ["cells", "faces"])
        assert got.num == 4
        assert got.get("cells").tolist() == [0, 1, 2, 3]
        print("ok test_it_stacks_disjoint_arrays")

    def test_it_selects_only_the_named_indices() -> None:
        got = concat_index_array(a(), b(), ["faces"])
        assert got.fdnames == ["faces"] and got.num == 4
        print("ok test_it_selects_only_the_named_indices")

    def test_the_column_order_follows_fdnames() -> None:
        got = concat_index_array(a(), b(), ["faces", "cells"])
        assert got.fdnames == ["faces", "cells"]
        assert got.get("faces").tolist() == [3, 4, 5, 6]
        print("ok test_the_column_order_follows_fdnames")

    def test_a_duplicate_row_is_rejected_by_default() -> None:
        try:
            concat_index_array(a(), a(), ["cells", "faces"])
        except ValueError as error:
            assert "count them twice" in str(error)
            print("ok test_a_duplicate_row_is_rejected_by_default")
            return
        raise AssertionError("a repeated multi-index must be rejected")

    def test_remove_duplicates_gives_the_set_union() -> None:
        got = concat_index_array(a(), a(), ["cells", "faces"],
                                 remove_duplicates=True)
        assert got.num == 2
        print("ok test_remove_duplicates_gives_the_set_union")

    def test_the_check_can_be_turned_off() -> None:
        got = concat_index_array(a(), a(), ["cells", "faces"],
                                 check_unique=False)
        assert got.num == 4, "duplicates should survive when unchecked"
        print("ok test_the_check_can_be_turned_off")

    def test_a_partial_projection_can_duplicate() -> None:
        """Two distinct pairs can share a cell, so projecting onto cells alone
        legitimately repeats - which is why the check is opt-out."""
        left = IndexArray({"cells": [0, 0], "faces": [3, 4]})
        got = concat_index_array(left, b(), ["cells"], check_unique=False)
        assert got.get("cells").tolist() == [0, 0, 2, 3]
        print("ok test_a_partial_projection_can_duplicate")

    test_it_stacks_disjoint_arrays()
    test_it_selects_only_the_named_indices()
    test_the_column_order_follows_fdnames()
    test_a_duplicate_row_is_rejected_by_default()
    test_remove_duplicates_gives_the_set_union()
    test_the_check_can_be_turned_off()
    test_a_partial_projection_can_duplicate()
    print("\nAll tests passed!")
