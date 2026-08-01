"""A contraction of two index arrays (MRST ``utils/tensorAssembly/TensorProd.m``).

Multiplies two quantities and sums over chosen indices - Einstein summation,
written with names instead of subscripts. Four groups of names control it:

- **`mergefds`** - shared, matched, and *kept* in the result. A merge index runs
  alongside: the product is formed separately for each of its values.
- **`reducefds`** - shared, matched, and *summed away*. This is the contraction
  proper.
- **`reducefds1` / `reducefds2`** - belong to one array only and are summed
  away, for when a quantity carries an index the other side does not.

Everything else survives into the result. So `A(cells,faces) * B(faces,nodes)`
reducing on `faces` gives `C(cells,nodes)`, while merging on `faces` instead
gives `C(cells,faces,nodes)`.

`eval(A, B)` gathers both onto the joined rows, multiplies elementwise, and
scatter-adds onto the result - one pass, no intermediate dense tensor. The
result index array is computed for you unless `tbl3` is given, in which case the
supplied one is matched against the join.

**The join is over `reducefds + mergefds` together**, which is what makes merge
and reduce differ only in whether the name survives the final projection. Worth
knowing when reading MRST's `setup`: the two lists are concatenated into
`crossfds` at the top and separated again at the bottom.
"""

from typing import Optional, Sequence

import numpy as np
import scipy.sparse as sps

from pymrst.utils.tensor_assembly.cross_index_array import cross_index_array
from pymrst.utils.tensor_assembly.proj_index_array import proj_index_array
from pymrst.utils.tensor_assembly.tensor_map import TensorMap
from pymrst.utils.tensor_assembly.utils.replacefield import replacefield


class TensorProd:
    """A named contraction of two index arrays."""

    def __init__(self, tbl1=None, tbl2=None, tbl3=None,
                 reducefds: Optional[Sequence[str]] = None,
                 mergefds: Optional[Sequence[str]] = None,
                 reducefds1: Optional[Sequence[str]] = None,
                 reducefds2: Optional[Sequence[str]] = None,
                 replacefds1=None, replacefds2=None, replacefds3=None):
        self.tbl1, self.tbl2, self.tbl3 = tbl1, tbl2, tbl3
        self.reducefds = list(reducefds) if reducefds else []
        self.mergefds = list(mergefds) if mergefds else []
        self.reducefds1 = list(reducefds1) if reducefds1 else []
        self.reducefds2 = list(reducefds2) if reducefds2 else []
        self.replacefds1 = replacefds1
        self.replacefds2 = replacefds2
        self.replacefds3 = replacefds3
        self.pivottbl = None
        self.dispind1 = self.dispind2 = self.dispind3 = None
        self.issetup = False
        self.settbl3 = tbl3 is None

    def setup(self) -> "TensorProd":
        """Build the pivot join and the three dispatch index vectors."""
        if self.issetup:
            return self
        if self.tbl1 is None or self.tbl2 is None:
            raise ValueError("a tensor product needs both index arrays")

        tbl1 = replacefield(self.tbl1, self.replacefds1) if self.replacefds1 \
            else self.tbl1
        tbl2 = replacefield(self.tbl2, self.replacefds2) if self.replacefds2 \
            else self.tbl2

        crossfds = list(self.reducefds) + list(self.mergefds)
        names1, names2 = list(tbl1.fdnames), list(tbl2.fdnames)
        for group, names, which in ((crossfds, names1, "first"),
                                    (crossfds, names2, "second")):
            missing = [n for n in group if n not in names]
            if missing:
                raise ValueError(
                    f"{missing} must be in the {which} array, which has {names}")

        other1 = [n for n in names1 if n not in crossfds]
        other2 = [n for n in names2 if n not in crossfds]
        clash = sorted(set(other1) & set(other2))
        if clash:
            raise ValueError(
                f"{clash} appear in both arrays but are neither merged nor "
                f"reduced; the product has no rule for them")

        pivottbl, indstruct = cross_index_array(
            tbl1, tbl2, crossfds,
            pure_product=not crossfds) if crossfds else (
            cross_index_array(tbl1, tbl2, [], pure_product=True), None)
        if indstruct is None:
            raise ValueError(
                "nothing is merged or reduced, so this is an outer product "
                "with no contraction; give reducefds or mergefds")

        self.dispind1 = indstruct[0]["inds"]
        self.dispind2 = indstruct[1]["inds"]

        if self.tbl3 is None:
            keep1 = [n for n in tbl1.fdnames
                     if n not in crossfds and n not in self.reducefds1]
            keep2 = [n for n in tbl2.fdnames
                     if n not in crossfds and n not in self.reducefds2]
            tbl3, indstruct3 = proj_index_array(
                pivottbl, keep1 + keep2 + list(self.mergefds))
            self.dispind3 = indstruct3["inds"]
            if self.replacefds3:
                tbl3 = replacefield(tbl3, self.replacefds3)
            self.tbl3 = tbl3
        else:
            names3 = [n for n in self.tbl3.fdnames
                      if n not in self.reducefds1 and n not in self.reducefds2]
            self.dispind3 = TensorMap(from_tbl=self.tbl3, to_tbl=pivottbl,
                                      mergefds=names3).get_dispatch_ind()

        self.pivottbl = pivottbl
        self.issetup = True
        return self

    def eval(self, a, b):
        """Contract ``a`` (on ``tbl1``) with ``b`` (on ``tbl2``)."""
        self._require_setup()
        product = a[self.dispind1] * b[self.dispind2]
        if isinstance(product, np.ndarray) and product.dtype.kind in "fiub":
            return np.bincount(self.dispind3, weights=product,
                               minlength=self.tbl3.num)
        # AD or object arrays: the same sum expressed as a matrix product.
        gather = sps.csr_matrix(
            (np.ones(self.dispind3.size),
             (self.dispind3, np.arange(self.dispind3.size))),
            shape=(self.tbl3.num, self.pivottbl.num))
        return gather @ product

    def get_matrices(self):
        """The three matrices behind ``eval``.

        ``m1`` and ``m2`` gather from ``tbl1``/``tbl2`` onto the pivot rows;
        ``m3`` scatter-adds from the pivot rows onto ``tbl3``. So
        ``m3 @ ((m1 @ a) * (m2 @ b))`` is exactly ``eval(a, b)``.
        """
        self._require_setup()
        n = self.pivottbl.num
        return {
            "m1": _scatter(self.dispind1, n, self.tbl1.num).T,
            "m2": _scatter(self.dispind2, n, self.tbl2.num).T,
            "m3": _scatter(self.dispind3, n, self.tbl3.num),
        }

    def get_dispatch_ind(self):
        """``(dispind1, dispind2)`` - what a caller assembling by hand needs."""
        self.setup()
        return self.dispind1, self.dispind2

    def _require_setup(self):
        if not self.issetup:
            raise ValueError(
                "the tensor product is not set up; call setup() first")

    def __repr__(self) -> str:
        state = "set up" if self.issetup else "not set up"
        return (f"<TensorProd reduce={self.reducefds} merge={self.mergefds}, "
                f"{state}>")


def _scatter(inds, n, size) -> sps.csr_matrix:
    return sps.csr_matrix((np.ones(inds.size), (inds, np.arange(n))),
                          shape=(size, n))


if __name__ == "__main__":
    from pymrst.utils.tensor_assembly.index_array import IndexArray

    def cf():
        return IndexArray({"cells": [0, 0, 1], "faces": [3, 7, 7]})

    def fn():
        return IndexArray({"faces": [3, 7, 7], "nodes": [10, 20, 30]})

    def test_reducing_contracts_the_shared_index() -> None:
        """A(cells,faces) * B(faces,nodes) summed over faces."""
        p = TensorProd(tbl1=cf(), tbl2=fn(), reducefds=["faces"]).setup()
        assert sorted(p.tbl3.fdnames) == ["cells", "nodes"]
        got = p.eval(np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0]))
        assert got.shape == (p.tbl3.num,)
        print("ok test_reducing_contracts_the_shared_index")

    def test_merging_keeps_the_shared_index() -> None:
        p = TensorProd(tbl1=cf(), tbl2=fn(), mergefds=["faces"]).setup()
        assert sorted(p.tbl3.fdnames) == ["cells", "faces", "nodes"]
        print("ok test_merging_keeps_the_shared_index")

    def test_it_agrees_with_the_explicit_sum() -> None:
        """The definition, computed by hand from the index arrays."""
        p = TensorProd(tbl1=cf(), tbl2=fn(), reducefds=["faces"]).setup()
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([10.0, 20.0, 30.0])
        got = p.eval(a, b)

        want = {}
        for i, (c, f) in enumerate(cf().gets(["cells", "faces"]).tolist()):
            for j, (f2, n) in enumerate(fn().gets(["faces", "nodes"]).tolist()):
                if f == f2:
                    want[(c, n)] = want.get((c, n), 0.0) + a[i] * b[j]
        for row, value in zip(p.tbl3.gets(["cells", "nodes"]).tolist(), got):
            assert np.isclose(value, want[tuple(row)]), (row, value)
        assert len(want) == p.tbl3.num
        print("ok test_it_agrees_with_the_explicit_sum")

    def test_a_matrix_vector_product() -> None:
        """Reducing a (row, col) array against a (col,) vector is A @ x."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        entries = IndexArray({"row": [0, 0, 1, 1], "col": [0, 1, 0, 1]})
        vector = IndexArray({"col": [0, 1]})
        p = TensorProd(tbl1=entries, tbl2=vector, reducefds=["col"]).setup()
        x = np.array([5.0, 6.0])
        got = p.eval(A.ravel(), x)
        order = p.tbl3.get("row")
        assert np.allclose(got[np.argsort(order)], A @ x)
        print("ok test_a_matrix_vector_product")

    def test_merging_gives_a_row_wise_product() -> None:
        entries = IndexArray({"row": [0, 1], "col": [0, 1]})
        scale = IndexArray({"row": [0, 1]})
        p = TensorProd(tbl1=entries, tbl2=scale, mergefds=["row"]).setup()
        got = p.eval(np.array([2.0, 3.0]), np.array([10.0, 100.0]))
        assert sorted(got.tolist()) == [20.0, 300.0]
        print("ok test_merging_gives_a_row_wise_product")

    def test_reducefds1_sums_an_index_only_one_side_has() -> None:
        left = IndexArray({"cells": [0, 0, 1], "faces": [3, 7, 7]})
        right = IndexArray({"cells": [0, 1]})
        p = TensorProd(tbl1=left, tbl2=right, mergefds=["cells"],
                       reducefds1=["faces"]).setup()
        assert p.tbl3.fdnames == ["cells"]
        got = p.eval(np.array([1.0, 10.0, 100.0]), np.array([2.0, 3.0]))
        order = np.argsort(p.tbl3.get("cells"))
        assert got[order].tolist() == [22.0, 300.0]
        print("ok test_reducefds1_sums_an_index_only_one_side_has")

    def test_a_given_result_table_is_honoured() -> None:
        given = IndexArray({"cells": [1, 0]})
        left = IndexArray({"cells": [0, 0, 1], "faces": [3, 7, 7]})
        right = IndexArray({"cells": [0, 1]})
        p = TensorProd(tbl1=left, tbl2=right, tbl3=given, mergefds=["cells"],
                       reducefds1=["faces"]).setup()
        got = p.eval(np.array([1.0, 10.0, 100.0]), np.array([2.0, 3.0]))
        # tbl3 lists cell 1 first, so the answers come in that order.
        assert got.tolist() == [300.0, 22.0], got.tolist()
        print("ok test_a_given_result_table_is_honoured")

    def test_the_matrices_reproduce_eval() -> None:
        p = TensorProd(tbl1=cf(), tbl2=fn(), reducefds=["faces"]).setup()
        a, b = np.array([1.0, 2.0, 3.0]), np.array([10.0, 20.0, 30.0])
        m = p.get_matrices()
        # m1 and m2 gather onto the pivot rows; m3 scatter-adds off them.
        assert np.allclose(m["m3"] @ ((m["m1"] @ a) * (m["m2"] @ b)),
                           p.eval(a, b))
        print("ok test_the_matrices_reproduce_eval")

    def test_renames_apply_before_the_product() -> None:
        left = IndexArray({"cells": [0, 0, 1], "f": [3, 7, 7]})
        p = TensorProd(tbl1=left, tbl2=fn(), reducefds=["faces"],
                       replacefds1=("f", "faces")).setup()
        assert sorted(p.tbl3.fdnames) == ["cells", "nodes"]
        print("ok test_renames_apply_before_the_product")

    def test_setup_is_idempotent() -> None:
        p = TensorProd(tbl1=cf(), tbl2=fn(), reducefds=["faces"])
        first = p.setup().dispind3.copy()
        assert p.setup().dispind3.tolist() == first.tolist()
        print("ok test_setup_is_idempotent")

    def test_using_it_before_setup_is_rejected() -> None:
        p = TensorProd(tbl1=cf(), tbl2=fn(), reducefds=["faces"])
        try:
            p.eval(np.ones(3), np.ones(3))
        except ValueError as error:
            assert "not set up" in str(error)
            print("ok test_using_it_before_setup_is_rejected")
            return
        raise AssertionError("using an unprepared product must be rejected")

    def test_an_ambiguous_shared_name_is_rejected() -> None:
        a = IndexArray({"cells": [0], "faces": [3]})
        b = IndexArray({"cells": [0], "faces": [3]})
        try:
            TensorProd(tbl1=a, tbl2=b, reducefds=["cells"]).setup()
        except ValueError as error:
            assert "neither merged nor reduced" in str(error)
            print("ok test_an_ambiguous_shared_name_is_rejected")
            return
        raise AssertionError("an ambiguous shared name must be rejected")

    def test_a_missing_reduce_name_is_rejected() -> None:
        try:
            TensorProd(tbl1=cf(), tbl2=fn(), reducefds=["nodes"]).setup()
        except ValueError as error:
            assert "must be in the first array" in str(error)
            print("ok test_a_missing_reduce_name_is_rejected")
            return
        raise AssertionError("an unknown reduce name must be rejected")

    test_reducing_contracts_the_shared_index()
    test_merging_keeps_the_shared_index()
    test_it_agrees_with_the_explicit_sum()
    test_a_matrix_vector_product()
    test_merging_gives_a_row_wise_product()
    test_reducefds1_sums_an_index_only_one_side_has()
    test_a_given_result_table_is_honoured()
    test_the_matrices_reproduce_eval()
    test_renames_apply_before_the_product()
    test_setup_is_idempotent()
    test_using_it_before_setup_is_rejected()
    test_an_ambiguous_shared_name_is_rejected()
    test_a_missing_reduce_name_is_rejected()
    print("\nAll tests passed!")
