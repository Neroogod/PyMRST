"""Select and sort index columns (MRST ``utils/tensorAssembly/sortIndexArray.m``).

Keeps the named index columns, in the order given, and sorts the rows
lexicographically by them. Used to put two index arrays into a common order
before they are matched.

**MRST's `keepAllFields` option does nothing.** It builds a local `fds` from
`fieldnames(tbl)` - which on a classdef returns the *property* names (`inds`,
`fdnames`, `tblname`, ...), not the index names - and then never uses it: the
output names were assigned from the argument before the branch runs. Calling it
with `keepAllFields` set produces exactly the same array plus a warning that
reads "option not really tested yet!", which was verified against MATLAB. It is
not implemented here; if you want the other columns kept, ask for them by name.
"""

from typing import Sequence

import numpy as np

from pymrst.utils.tensor_assembly.index_array import IndexArray


def sort_index_array(tbl, fds: Sequence[str]) -> IndexArray:
    """Return ``tbl`` reduced to ``fds`` and sorted by them."""
    fds = list(fds)
    inds = tbl.gets(fds)
    if inds.shape[0] > 1:
        order = np.lexsort(tuple(inds[:, i] for i in reversed(range(inds.shape[1]))))
        inds = inds[order]
    return IndexArray(fdnames=fds, inds=inds)


sortIndexArray = sort_index_array


if __name__ == "__main__":

    def table():
        return IndexArray({"cells": [2, 1, 2], "faces": [5, 4, 3]})

    def test_it_sorts_by_the_named_column() -> None:
        got = sort_index_array(table(), ["cells", "faces"])
        assert got.gets(["cells", "faces"]).tolist() == [[1, 4], [2, 3], [2, 5]]
        print("ok test_it_sorts_by_the_named_column")

    def test_the_first_name_is_the_primary_key() -> None:
        got = sort_index_array(table(), ["faces", "cells"])
        assert got.get("faces").tolist() == [3, 4, 5]
        print("ok test_the_first_name_is_the_primary_key")

    def test_it_keeps_only_what_was_asked_for() -> None:
        got = sort_index_array(table(), ["cells"])
        assert got.fdnames == ["cells"] and got.get("cells").tolist() == [1, 2, 2]
        print("ok test_it_keeps_only_what_was_asked_for")

    def test_the_original_is_untouched() -> None:
        t = table()
        sort_index_array(t, ["cells"])
        assert t.get("cells").tolist() == [2, 1, 2]
        print("ok test_the_original_is_untouched")

    def test_rows_stay_together() -> None:
        """Sorting must permute whole rows, not columns independently."""
        got = sort_index_array(table(), ["cells", "faces"])
        pairs = {tuple(r) for r in got.gets(["cells", "faces"]).tolist()}
        assert pairs == {(2, 5), (1, 4), (2, 3)}
        print("ok test_rows_stay_together")

    def test_a_single_row_is_fine() -> None:
        got = sort_index_array(IndexArray({"cells": [7]}), ["cells"])
        assert got.get("cells").tolist() == [7]
        print("ok test_a_single_row_is_fine")

    def test_an_unknown_name_is_rejected() -> None:
        try:
            sort_index_array(table(), ["nodes"])
        except ValueError as error:
            assert "no index named" in str(error)
            print("ok test_an_unknown_name_is_rejected")
            return
        raise AssertionError("an unknown index name must be rejected")

    test_it_sorts_by_the_named_column()
    test_the_first_name_is_the_primary_key()
    test_it_keeps_only_what_was_asked_for()
    test_the_original_is_untouched()
    test_rows_stay_together()
    test_a_single_row_is_fine()
    test_an_unknown_name_is_rejected()
    print("\nAll tests passed!")
