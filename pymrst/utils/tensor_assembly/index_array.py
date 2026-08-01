"""Named multi-index arrays (MRST ``utils/tensorAssembly/IndexArray.m``).

An `IndexArray` is an `(N, D)` integer table whose D columns each carry a
*name*: cells, faces, nodes, a spatial direction. It describes a subspace of a
tensor space by listing the multi-indices that belong to it - the rows of

    cells  faces
      0      3
      0      7
      1      7

say the subspace contains those three (cell, face) pairs and no others. Every
other class here (`TensorMap`, `TensorProd`, `SparseTensor`) is built on it:
they match, contract and reduce along index *names* rather than positions,
which is what makes an assembly readable instead of a nest of `reshape` calls.

**Value semantics.** MRST's is a plain `classdef`, so `tbl = tbl.addInd(...)`
returns a modified copy and leaves the original alone. The methods here do the
same and return new instances, rather than mutating in place - code ported from
MRST would otherwise silently share state.

**Indices are 0-based**, as everywhere in PyMRST and unlike MRST. That matters
most for `add_loc_ind`, which numbers the rows: MRST gives `1..N`, this gives
`0..N-1`, and the result is normally used to index straight into an array.

A **virtual** array keeps its names and its size but not its indices. MRST uses
this when a caller manages the local numbering itself; the rows do not exist,
so anything that reads them raises rather than inventing them.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


class IndexArray:
    """An ``(N, D)`` table of multi-indices with one name per column."""

    def __init__(self, structtbl: Optional[Dict] = None, *,
                 fdnames: Optional[Sequence[str]] = None,
                 inds=None, isvirtual: bool = False,
                 num: Optional[int] = None, tblname: Optional[str] = None):
        self.tblname = tblname
        self.isvirtual = bool(isvirtual)
        self.vnum = None
        self.parents = None

        if structtbl:
            names = list(structtbl)
            columns = [np.asarray(structtbl[n], dtype=np.int64).reshape(-1)
                       for n in names]
            lengths = {c.size for c in columns}
            if len(lengths) > 1:
                raise ValueError(
                    f"every index vector must have the same length, got "
                    f"{ {n: c.size for n, c in zip(names, columns)} }")
            self._setup(names, np.column_stack(columns) if columns
                        else np.zeros((0, 0), dtype=np.int64))
        else:
            self._setup(list(fdnames or []),
                        np.zeros((0, len(fdnames or [])), dtype=np.int64)
                        if inds is None else np.asarray(inds, dtype=np.int64))

        if self.isvirtual:
            if num is None:
                raise ValueError(
                    "a virtual index array has no rows to count, so its size "
                    "must be given as `num`")
            self.vnum = int(num)

    # ------------------------------------------------------------------ core

    def _setup(self, fdnames, inds):
        if len(set(fdnames)) != len(fdnames):
            duplicated = sorted({n for n in fdnames if fdnames.count(n) > 1})
            raise ValueError(f"repeated index names: {duplicated}")
        inds = np.asarray(inds, dtype=np.int64)
        if inds.ndim == 1:
            inds = inds.reshape(-1, 1)
        if inds.size and inds.shape[1] != len(fdnames):
            raise ValueError(
                f"{inds.shape[1]} index columns but {len(fdnames)} names")
        self.fdnames = list(fdnames)
        self.inds = inds

    def _copy(self, fdnames, inds) -> "IndexArray":
        out = IndexArray(fdnames=list(fdnames), inds=inds,
                         tblname=self.tblname)
        out.parents = self.parents
        return out

    def _column(self, name: str) -> int:
        try:
            return self.fdnames.index(name)
        except ValueError:
            raise ValueError(
                f"no index named {name!r}; this array has {self.fdnames}"
            ) from None

    def _require_real(self, what: str):
        if self.isvirtual:
            raise ValueError(
                f"cannot {what} a virtual index array - it carries names and a "
                f"size but no indices")

    # --------------------------------------------------------------- queries

    @property
    def num(self) -> int:
        """Number of multi-indices."""
        return self.vnum if self.isvirtual else int(self.inds.shape[0])

    def get(self, fdname: str) -> np.ndarray:
        """The index vector named ``fdname``."""
        self._require_real("read from")
        return self.inds[:, self._column(fdname)]

    def gets(self, getfdnames: Sequence[str]) -> np.ndarray:
        """The index vectors named ``getfdnames``, as columns."""
        self._require_real("read from")
        return self.inds[:, [self._column(n) for n in getfdnames]]

    # -------------------------------------------------------------- reshaping

    def add_ind(self, fdname: str, ind) -> "IndexArray":
        """Return a copy with one more named index column."""
        self._require_real("add an index to")
        if fdname in self.fdnames:
            raise ValueError(f"there is already an index named {fdname!r}")
        ind = np.asarray(ind, dtype=np.int64).reshape(-1, 1)
        if ind.shape[0] != self.num:
            raise ValueError(
                f"the new index has {ind.shape[0]} entries but the array has "
                f"{self.num} rows")
        stacked = (ind if self.inds.size == 0 and not self.fdnames
                   else np.hstack([self.inds, ind]))
        return self._copy(self.fdnames + [fdname], stacked)

    def add_loc_ind(self, locindname: str) -> "IndexArray":
        """Return a copy with a column numbering the rows, **0-based**."""
        return self.add_ind(locindname, np.arange(self.num, dtype=np.int64))

    def remove_ind(self, rmfdnames) -> "IndexArray":
        """Return a copy without the named index columns."""
        self._require_real("remove an index from")
        if isinstance(rmfdnames, str):
            rmfdnames = [rmfdnames]
        keep = list(range(len(self.fdnames)))
        for name in rmfdnames:
            keep.remove(self._column(name))
        return self._copy([self.fdnames[i] for i in keep], self.inds[:, keep])

    def duplicate_ind(self, fdcell) -> "IndexArray":
        """Split one index column into two identical ones, renamed.

        ``duplicate_ind(("cells", ("c1", "c2")))`` replaces `cells` with two
        copies of itself called `c1` and `c2`, in place. That is how a
        symmetric pairing - cell against cell - is expressed before contracting
        one side away.
        """
        self._require_real("duplicate an index of")
        old, (first, second) = fdcell
        at = self._column(old)
        order = list(range(at + 1)) + list(range(at, len(self.fdnames)))
        names = [self.fdnames[i] for i in order]
        names[at], names[at + 1] = first, second
        return self._copy(names, self.inds[:, order])

    # ---------------------------------------------------------------- display

    def __repr__(self) -> str:
        kind = "virtual " if self.isvirtual else ""
        name = f" {self.tblname!r}" if self.tblname else ""
        return f"<{kind}IndexArray{name} {self.num}x{len(self.fdnames)} {self.fdnames}>"

    def print(self, rows=None, fdnames=None) -> str:
        """A readable table; returned as well as printed, so tests can assert."""
        if self.isvirtual:
            text = f"virtual, {self.num} rows: {' '.join(self.fdnames)}"
            print(text)
            return text
        names = list(self.fdnames) if fdnames is None else list(fdnames)
        columns = [self._column(n) for n in names]
        chosen = self.inds if rows is None else self.inds[np.asarray(rows)]
        lines = ["  ".join(f"{n:>8}" for n in names)]
        lines += ["  ".join(f"{v:8d}" for v in row)
                  for row in chosen[:, columns]]
        text = "\n".join(lines)
        print(text)
        return text


if __name__ == "__main__":

    def table():
        return IndexArray({"cells": [0, 0, 1], "faces": [3, 7, 7]})

    def test_it_is_built_from_named_columns() -> None:
        t = table()
        assert t.num == 3 and t.fdnames == ["cells", "faces"]
        assert t.get("faces").tolist() == [3, 7, 7]
        print("ok test_it_is_built_from_named_columns")

    def test_gets_returns_columns_in_the_order_asked() -> None:
        assert table().gets(["faces", "cells"]).tolist() == [[3, 0], [7, 0], [7, 1]]
        print("ok test_gets_returns_columns_in_the_order_asked")

    def test_adding_an_index_does_not_touch_the_original() -> None:
        """Value semantics, as in MRST; mutating would silently share state."""
        t = table()
        more = t.add_ind("dir", [0, 1, 2])
        assert more.fdnames == ["cells", "faces", "dir"]
        assert t.fdnames == ["cells", "faces"], "the original was modified"
        print("ok test_adding_an_index_does_not_touch_the_original")

    def test_the_local_index_is_zero_based() -> None:
        """MRST numbers 1..N; PyMRST indexes arrays directly, so 0..N-1."""
        assert table().add_loc_ind("k").get("k").tolist() == [0, 1, 2]
        print("ok test_the_local_index_is_zero_based")

    def test_removing_an_index() -> None:
        t = table().remove_ind(["faces"])
        assert t.fdnames == ["cells"] and t.inds.shape == (3, 1)
        print("ok test_removing_an_index")

    def test_duplicating_an_index_splits_it_in_place() -> None:
        t = table().duplicate_ind(("cells", ("c1", "c2")))
        assert t.fdnames == ["c1", "c2", "faces"]
        assert t.get("c1").tolist() == t.get("c2").tolist() == [0, 0, 1]
        print("ok test_duplicating_an_index_splits_it_in_place")

    def test_a_virtual_array_keeps_names_and_size_only() -> None:
        v = IndexArray(fdnames=["cells"], isvirtual=True, num=17)
        assert v.num == 17 and v.fdnames == ["cells"]
        print("ok test_a_virtual_array_keeps_names_and_size_only")

    def test_reading_a_virtual_array_is_rejected() -> None:
        """It has no rows; inventing them would be worse than stopping."""
        v = IndexArray(fdnames=["cells"], isvirtual=True, num=17)
        try:
            v.get("cells")
        except ValueError as error:
            assert "virtual index array" in str(error)
            print("ok test_reading_a_virtual_array_is_rejected")
            return
        raise AssertionError("reading a virtual array must be rejected")

    def test_a_virtual_array_needs_its_size() -> None:
        try:
            IndexArray(fdnames=["cells"], isvirtual=True)
        except ValueError as error:
            assert "must be given as `num`" in str(error)
            print("ok test_a_virtual_array_needs_its_size")
            return
        raise AssertionError("a virtual array without num must be rejected")

    def test_repeated_names_are_rejected() -> None:
        try:
            IndexArray(fdnames=["cells", "cells"], inds=np.zeros((2, 2)))
        except ValueError as error:
            assert "repeated index names" in str(error)
            print("ok test_repeated_names_are_rejected")
            return
        raise AssertionError("duplicate names must be rejected")

    def test_ragged_columns_are_rejected() -> None:
        try:
            IndexArray({"cells": [0, 1], "faces": [3]})
        except ValueError as error:
            assert "same length" in str(error)
            print("ok test_ragged_columns_are_rejected")
            return
        raise AssertionError("ragged input must be rejected")

    def test_an_unknown_name_says_what_is_there() -> None:
        try:
            table().get("nodes")
        except ValueError as error:
            assert "cells" in str(error) and "faces" in str(error)
            print("ok test_an_unknown_name_says_what_is_there")
            return
        raise AssertionError("an unknown index name must be rejected")

    def test_adding_a_name_twice_is_rejected() -> None:
        try:
            table().add_ind("cells", [1, 2, 3])
        except ValueError as error:
            assert "already an index named" in str(error)
            print("ok test_adding_a_name_twice_is_rejected")
            return
        raise AssertionError("a repeated name must be rejected")

    def test_a_mismatched_column_length_is_rejected() -> None:
        try:
            table().add_ind("dir", [0, 1])
        except ValueError as error:
            assert "but the array has 3 rows" in str(error)
            print("ok test_a_mismatched_column_length_is_rejected")
            return
        raise AssertionError("a short column must be rejected")

    def test_print_is_readable_and_returnable() -> None:
        text = table().print()
        assert "cells" in text and "faces" in text
        assert len(text.splitlines()) == 4          # header plus three rows
        print("ok test_print_is_readable_and_returnable")

    test_it_is_built_from_named_columns()
    test_gets_returns_columns_in_the_order_asked()
    test_adding_an_index_does_not_touch_the_original()
    test_the_local_index_is_zero_based()
    test_removing_an_index()
    test_duplicating_an_index_splits_it_in_place()
    test_a_virtual_array_keeps_names_and_size_only()
    test_reading_a_virtual_array_is_rejected()
    test_a_virtual_array_needs_its_size()
    test_repeated_names_are_rejected()
    test_ragged_columns_are_rejected()
    test_an_unknown_name_says_what_is_there()
    test_adding_a_name_twice_is_rejected()
    test_a_mismatched_column_length_is_rejected()
    test_print_is_readable_and_returnable()
    print("\nAll tests passed!")
