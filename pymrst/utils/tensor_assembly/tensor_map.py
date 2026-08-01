"""A linear map between index arrays (MRST ``utils/tensorAssembly/TensorMap.m``).

Describes how a quantity indexed by `from_tbl` is carried to one indexed by
`to_tbl`, matching on the index names they share. Applying it does two things
at once - **gather** the source entries onto the joined rows, then **scatter-add**
them onto the destination - which between them express dispatch, summation and
any mixture of the two:

- one source row per destination row is a **permutation** or a lookup;
- many sources per destination **sums** them, e.g. cell-face values onto cells;
- one source per many destinations **broadcasts**, e.g. a cell value onto each
  of its faces.

Which of those you get follows from the index arrays, not from a flag. That is
the point of the abstraction: `mergefds` says what lines up with what, and the
arithmetic follows.

`setup()` computes the join once and caches the two dispatch index vectors;
`eval` and `eval_transpose` are then just a gather and a `bincount`. The
transpose is the adjoint - summation one way is broadcast the other - which is
what makes these usable in an assembled Jacobian.

**MRST's "given pivot table" branch cannot have run.** It reads
`instruct{1}.inds` (for `indstruct`), calls `crossIndexArray(pivotbl,
givenpivotbl, ...)` with two more misspellings, and uses `indi2` before
assigning it. Four errors in nine lines, any one of which raises. The port
implements the evident intent - reorder the dispatch indices to match a
caller-supplied pivot - and says so here rather than reproducing dead code.
"""

from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sps

from pymrst.utils.tensor_assembly.cross_index_array import cross_index_array
from pymrst.utils.tensor_assembly.utils.replacefield import replacefield


class TensorMap:
    """A gather-then-scatter map from ``from_tbl`` to ``to_tbl``."""

    def __init__(self, from_tbl=None, to_tbl=None,
                 mergefds: Optional[Sequence[str]] = None,
                 replace_from_tbl_fds=None, replace_to_tbl_fds=None,
                 pivottbl=None):
        self.fromTbl = from_tbl
        self.toTbl = to_tbl
        self.mergefds = list(mergefds) if mergefds else []
        self.replaceFromTblfds = replace_from_tbl_fds
        self.replaceToTblfds = replace_to_tbl_fds
        self.pivottbl = pivottbl
        self.dispind1 = None
        self.dispind2 = None
        self.issetup = False

    # ------------------------------------------------------------------ setup

    def setup(self) -> "TensorMap":
        """Compute the pivot join and the two dispatch index vectors."""
        if self.issetup:
            return self

        from_tbl = (replacefield(self.fromTbl, self.replaceFromTblfds)
                    if self.replaceFromTblfds else self.fromTbl)
        to_tbl = (replacefield(self.toTbl, self.replaceToTblfds)
                  if self.replaceToTblfds else self.toTbl)
        if from_tbl is None or to_tbl is None:
            raise ValueError("a tensor map needs both index arrays")

        names1, names2 = list(from_tbl.fdnames), list(to_tbl.fdnames)
        for name in self.mergefds:
            if name not in names1 or name not in names2:
                raise ValueError(
                    f"merge name {name!r} must be in both arrays; they have "
                    f"{names1} and {names2}")

        # A name shared but not merged would be ambiguous: it is neither a key
        # nor reduced, so the join has no rule for it.
        other1 = [n for n in names1 if n not in self.mergefds]
        other2 = [n for n in names2 if n not in self.mergefds]
        clash = sorted(set(other1) & set(other2))
        if clash:
            raise ValueError(
                f"{clash} appear in both arrays but are neither merged nor "
                f"reduced; add them to mergefds or rename one side")

        given = self.pivottbl
        pivottbl, indstruct = cross_index_array(
            from_tbl, to_tbl, self.mergefds,
            pure_product=not self.mergefds) if self.mergefds else (
            cross_index_array(from_tbl, to_tbl, [], pure_product=True), None)

        if indstruct is None:
            raise ValueError(
                "a tensor map with no merge names is a full outer product, "
                "which has no dispatch indices; give mergefds")

        dispind1 = indstruct[0]["inds"]
        dispind2 = indstruct[1]["inds"]

        if given is not None:
            # Same multi-indices, possibly in another order: line ours up.
            if sorted(given.fdnames) != sorted(pivottbl.fdnames):
                raise ValueError(
                    f"the given pivot has names {given.fdnames}, the join "
                    f"produces {pivottbl.fdnames}")
            order = _match_rows(pivottbl, given)
            dispind1, dispind2 = dispind1[order], dispind2[order]
            pivottbl = given

        self.pivottbl = pivottbl
        self.dispind1, self.dispind2 = dispind1, dispind2
        self.issetup = True
        return self

    # ------------------------------------------------------------ application

    def eval(self, u):
        """Carry ``u`` (indexed by ``from_tbl``) to ``to_tbl``."""
        self._require_setup()
        gathered = u[self.dispind1]
        if isinstance(gathered, np.ndarray) and gathered.dtype.kind in "fiub":
            return np.bincount(self.dispind2, weights=gathered,
                               minlength=self.toTbl.num)
        # AD or object arrays: express the same thing as a matrix product.
        return self.matrix() @ u

    def eval_transpose(self, u):
        """The adjoint: carry ``u`` (indexed by ``to_tbl``) back to ``from_tbl``."""
        self._require_setup()
        gathered = u[self.dispind2]
        if isinstance(gathered, np.ndarray) and gathered.dtype.kind in "fiub":
            return np.bincount(self.dispind1, weights=gathered,
                               minlength=self.fromTbl.num)
        return self.matrix().T @ u

    def matrix(self) -> sps.csr_matrix:
        """The map as a sparse matrix, for AD or for assembly."""
        self._require_setup()
        return sps.csr_matrix(
            (np.ones(self.dispind1.size), (self.dispind2, self.dispind1)),
            shape=(self.toTbl.num, self.fromTbl.num))

    def get_dispatch_ind(self) -> np.ndarray:
        """For each ``to_tbl`` row, which ``from_tbl`` row reaches it.

        Only meaningful when the map is one-to-one onto `to_tbl`; if several
        source rows land on one destination there is no single answer, and this
        raises rather than picking one.
        """
        self.setup()
        counts = np.bincount(self.dispind2, minlength=self.toTbl.num)
        if np.any(counts != 1):
            raise ValueError(
                f"{int(np.count_nonzero(counts != 1))} destination rows do not "
                f"receive exactly one source row, so there is no single "
                f"dispatch index; use eval() to sum them instead")
        inverse = np.empty(self.toTbl.num, dtype=np.int64)
        inverse[self.dispind2] = np.arange(self.dispind2.size)
        return self.dispind1[inverse]

    def _require_setup(self):
        if not self.issetup:
            raise ValueError("the tensor map is not set up; call setup() first")

    def __repr__(self) -> str:
        state = "set up" if self.issetup else "not set up"
        return f"<TensorMap on {self.mergefds}, {state}>"


def _match_rows(pivottbl, given) -> np.ndarray:
    """Row permutation taking ``pivottbl`` onto ``given``."""
    names = list(given.fdnames)
    ours = {tuple(row): i for i, row in enumerate(pivottbl.gets(names).tolist())}
    try:
        return np.array([ours[tuple(row)] for row in given.gets(names).tolist()],
                        dtype=np.int64)
    except KeyError as missing:
        raise ValueError(
            f"the given pivot table has a row {missing.args[0]} the join does "
            f"not produce") from None


if __name__ == "__main__":
    from pymrst.utils.tensor_assembly.index_array import IndexArray

    def cell_face():
        # cell 0 has faces 3 and 7; cell 1 has face 7.
        return IndexArray({"cells": [0, 0, 1], "faces": [3, 7, 7]})

    def cells():
        return IndexArray({"cells": [0, 1]})

    def test_it_sums_onto_the_coarser_index() -> None:
        m = TensorMap(from_tbl=cell_face(), to_tbl=cells(),
                      mergefds=["cells"]).setup()
        got = m.eval(np.array([1.0, 10.0, 100.0]))
        assert got.tolist() == [11.0, 100.0], got
        print("ok test_it_sums_onto_the_coarser_index")

    def test_it_broadcasts_the_other_way() -> None:
        m = TensorMap(from_tbl=cells(), to_tbl=cell_face(),
                      mergefds=["cells"]).setup()
        got = m.eval(np.array([5.0, 9.0]))
        assert got.tolist() == [5.0, 5.0, 9.0], got
        print("ok test_it_broadcasts_the_other_way")

    def test_the_transpose_is_the_adjoint() -> None:
        """<M u, v> == <u, M' v> for any u and v."""
        m = TensorMap(from_tbl=cell_face(), to_tbl=cells(),
                      mergefds=["cells"]).setup()
        rng = np.random.default_rng(0)
        u, v = rng.normal(size=3), rng.normal(size=2)
        assert np.isclose(m.eval(u) @ v, u @ m.eval_transpose(v))
        print("ok test_the_transpose_is_the_adjoint")

    def test_summing_then_broadcasting_counts_the_faces() -> None:
        down = TensorMap(from_tbl=cell_face(), to_tbl=cells(),
                         mergefds=["cells"]).setup()
        assert down.eval(np.ones(3)).tolist() == [2.0, 1.0]
        print("ok test_summing_then_broadcasting_counts_the_faces")

    def test_the_matrix_agrees_with_eval() -> None:
        m = TensorMap(from_tbl=cell_face(), to_tbl=cells(),
                      mergefds=["cells"]).setup()
        u = np.array([1.0, 10.0, 100.0])
        assert np.allclose(m.matrix() @ u, m.eval(u))
        print("ok test_the_matrix_agrees_with_eval")

    def test_a_permutation_is_recovered_exactly() -> None:
        a = IndexArray({"cells": [0, 1, 2]})
        b = IndexArray({"cells": [2, 0, 1]})
        m = TensorMap(from_tbl=a, to_tbl=b, mergefds=["cells"]).setup()
        assert m.eval(np.array([10.0, 20.0, 30.0])).tolist() == [30.0, 10.0, 20.0]
        print("ok test_a_permutation_is_recovered_exactly")

    def test_get_dispatch_ind_for_a_one_to_one_map() -> None:
        a = IndexArray({"cells": [0, 1, 2]})
        b = IndexArray({"cells": [2, 0, 1]})
        m = TensorMap(from_tbl=a, to_tbl=b, mergefds=["cells"])
        assert m.get_dispatch_ind().tolist() == [2, 0, 1]
        print("ok test_get_dispatch_ind_for_a_one_to_one_map")

    def test_get_dispatch_ind_refuses_a_many_to_one_map() -> None:
        """There is no single source index when several are summed."""
        m = TensorMap(from_tbl=cell_face(), to_tbl=cells(), mergefds=["cells"])
        try:
            m.get_dispatch_ind()
        except ValueError as error:
            assert "exactly one source row" in str(error)
            print("ok test_get_dispatch_ind_refuses_a_many_to_one_map")
            return
        raise AssertionError("a many-to-one map has no dispatch index")

    def test_renames_apply_before_the_join() -> None:
        a = IndexArray({"c": [0, 0, 1], "faces": [3, 7, 7]})
        m = TensorMap(from_tbl=a, to_tbl=cells(), mergefds=["cells"],
                      replace_from_tbl_fds=("c", "cells")).setup()
        assert m.eval(np.array([1.0, 10.0, 100.0])).tolist() == [11.0, 100.0]
        print("ok test_renames_apply_before_the_join")

    def test_setup_is_idempotent() -> None:
        m = TensorMap(from_tbl=cell_face(), to_tbl=cells(), mergefds=["cells"])
        before = m.setup().dispind1.copy()
        assert m.setup().dispind1.tolist() == before.tolist()
        print("ok test_setup_is_idempotent")

    def test_using_it_before_setup_is_rejected() -> None:
        m = TensorMap(from_tbl=cell_face(), to_tbl=cells(), mergefds=["cells"])
        try:
            m.eval(np.ones(3))
        except ValueError as error:
            assert "not set up" in str(error)
            print("ok test_using_it_before_setup_is_rejected")
            return
        raise AssertionError("using an unprepared map must be rejected")

    def test_an_ambiguous_shared_name_is_rejected() -> None:
        """A name in both arrays that is neither merged nor reduced."""
        a = IndexArray({"cells": [0], "faces": [3]})
        b = IndexArray({"cells": [0], "faces": [3]})
        try:
            TensorMap(from_tbl=a, to_tbl=b, mergefds=["cells"]).setup()
        except ValueError as error:
            assert "neither merged nor reduced" in str(error)
            print("ok test_an_ambiguous_shared_name_is_rejected")
            return
        raise AssertionError("an ambiguous shared name must be rejected")

    def test_an_unknown_merge_name_is_rejected() -> None:
        try:
            TensorMap(from_tbl=cell_face(), to_tbl=cells(),
                      mergefds=["nodes"]).setup()
        except ValueError as error:
            assert "must be in both arrays" in str(error)
            print("ok test_an_unknown_merge_name_is_rejected")
            return
        raise AssertionError("an unknown merge name must be rejected")

    test_it_sums_onto_the_coarser_index()
    test_it_broadcasts_the_other_way()
    test_the_transpose_is_the_adjoint()
    test_summing_then_broadcasting_counts_the_faces()
    test_the_matrix_agrees_with_eval()
    test_a_permutation_is_recovered_exactly()
    test_get_dispatch_ind_for_a_one_to_one_map()
    test_get_dispatch_ind_refuses_a_many_to_one_map()
    test_renames_apply_before_the_join()
    test_setup_is_idempotent()
    test_using_it_before_setup_is_rejected()
    test_an_ambiguous_shared_name_is_rejected()
    test_an_unknown_merge_name_is_rejected()
    print("\nAll tests passed!")
