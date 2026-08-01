"""Rename, swap or drop index names (MRST ``tensorAssembly/utils/replacefield.m``).

Three operations, chosen by the shape of each pair:

    ("cells", "c")                  rename `cells` to `c`
    ("cells", "")                   drop the `cells` column entirely
    ("cells", "faces", "interchange")   swap the two names

Renaming is how an index array is made to line up with another before a
contraction: the same physical index goes by different names in different
tables, and `TensorMap` matches on names.

**Dropping a column is not a projection.** Removing `cells` from a table that
lists (cell, face) pairs leaves the face column with one row per original pair,
including repeats - it does not reduce to distinct faces. MRST's comment says
only "we erase this field", and the difference matters enough to state.
"""

from typing import Sequence

from pymrst.utils.tensor_assembly.index_array import IndexArray


def replacefield(tbl, fieldpairs) -> IndexArray:
    """Apply one or several rename/drop/interchange pairs, in order."""
    pairs = ([fieldpairs] if fieldpairs and isinstance(fieldpairs[0], str)
             else list(fieldpairs))
    for pair in pairs:
        tbl = _replace_one(tbl, pair)
    return tbl


def _replace_one(tbl, fieldpair: Sequence[str]) -> IndexArray:
    old, new = fieldpair[0], fieldpair[1]
    if len(fieldpair) > 2:
        if fieldpair[2] != "interchange":
            raise ValueError(
                f"the third element must be 'interchange', got {fieldpair[2]!r}")
        first, second = tbl._column(old), tbl._column(new)
        names = list(tbl.fdnames)
        names[first], names[second] = names[second], names[first]
        return tbl._copy(names, tbl.inds)

    at = tbl._column(old)
    if new == "":
        return tbl.remove_ind([old])
    if new in tbl.fdnames:
        raise ValueError(
            f"renaming {old!r} to {new!r} would duplicate an existing index "
            f"name; use ('{old}', '{new}', 'interchange') to swap them")
    names = list(tbl.fdnames)
    names[at] = new
    return tbl._copy(names, tbl.inds)


if __name__ == "__main__":

    def table():
        return IndexArray({"cells": [0, 1], "faces": [3, 4]})

    def test_it_renames_an_index() -> None:
        got = replacefield(table(), ("cells", "c"))
        assert got.fdnames == ["c", "faces"]
        assert got.get("c").tolist() == [0, 1]
        print("ok test_it_renames_an_index")

    def test_renaming_leaves_the_original_alone() -> None:
        t = table()
        replacefield(t, ("cells", "c"))
        assert t.fdnames == ["cells", "faces"]
        print("ok test_renaming_leaves_the_original_alone")

    def test_several_pairs_apply_in_order() -> None:
        got = replacefield(table(), [("cells", "c"), ("faces", "f")])
        assert got.fdnames == ["c", "f"]
        print("ok test_several_pairs_apply_in_order")

    def test_an_empty_name_drops_the_column() -> None:
        got = replacefield(table(), ("cells", ""))
        assert got.fdnames == ["faces"] and got.inds.shape == (2, 1)
        print("ok test_an_empty_name_drops_the_column")

    def test_dropping_is_not_a_projection() -> None:
        """The row count is unchanged, repeats and all; see the docstring."""
        t = IndexArray({"cells": [0, 0], "faces": [3, 4]})
        got = replacefield(t, ("faces", ""))
        assert got.num == 2 and got.get("cells").tolist() == [0, 0]
        print("ok test_dropping_is_not_a_projection")

    def test_interchange_swaps_two_names() -> None:
        got = replacefield(table(), ("cells", "faces", "interchange"))
        assert got.fdnames == ["faces", "cells"]
        # The columns did not move, so the values follow the new names.
        assert got.get("faces").tolist() == [0, 1]
        assert got.get("cells").tolist() == [3, 4]
        print("ok test_interchange_swaps_two_names")

    def test_renaming_onto_an_existing_name_is_rejected() -> None:
        try:
            replacefield(table(), ("cells", "faces"))
        except ValueError as error:
            assert "interchange" in str(error)
            print("ok test_renaming_onto_an_existing_name_is_rejected")
            return
        raise AssertionError("a colliding rename must be rejected")

    def test_an_unknown_name_is_rejected() -> None:
        try:
            replacefield(table(), ("nodes", "n"))
        except ValueError as error:
            assert "no index named" in str(error)
            print("ok test_an_unknown_name_is_rejected")
            return
        raise AssertionError("an unknown index must be rejected")

    def test_a_bad_third_element_is_rejected() -> None:
        try:
            replacefield(table(), ("cells", "faces", "swap"))
        except ValueError as error:
            assert "'interchange'" in str(error)
            print("ok test_a_bad_third_element_is_rejected")
            return
        raise AssertionError("an unknown modifier must be rejected")

    test_it_renames_an_index()
    test_renaming_leaves_the_original_alone()
    test_several_pairs_apply_in_order()
    test_an_empty_name_drops_the_column()
    test_dropping_is_not_a_projection()
    test_interchange_swaps_two_names()
    test_renaming_onto_an_existing_name_is_rejected()
    test_an_unknown_name_is_rejected()
    test_a_bad_third_element_is_rejected()
    print("\nAll tests passed!")
