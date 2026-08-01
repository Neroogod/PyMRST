"""A sparse operator between index arrays (MRST ``.../SparseTensor.m``).

Holds a sparse matrix in coordinate form - `tensval` at `(row, col)` - together
with the index arrays naming its domain and range. It is what a `TensorMap` or
a partially-applied `TensorProd` becomes once you want to multiply, add or
transpose it like an operator.

Two ways in:

- `from_tensor_map(map)` - the map's own matrix, all ones. Dispatch and
  summation as an operator.
- `from_tensor_prod(vec, prod, argindex)` - **fix one argument of a product**
  and keep the other free. With `argindex=1`, `vec` is the first argument and
  the result maps `tbl2` to `tbl3`. This is how a coefficient field becomes the
  operator it induces, which is the usual reason to build one of these during
  assembly.

`matmul` composes two of them. MRST computes it either from the cached matrices
or, when those are absent, by re-entering its own machinery: it makes index
arrays of the two coordinate lists, joins them on the shared inner index, and
projects onto the outer pair - matrix multiplication expressed in the same
algebra. Both give the same answer; this uses SciPy's, and a test below checks
it against the index-array route so the equivalence is asserted rather than
assumed.

**Duplicate coordinates sum**, as they do in every sparse format, which is what
makes `from_tensor_prod` correct when several pivot rows land on one entry.
"""

from typing import Optional

import numpy as np
import scipy.sparse as sps


class SparseTensor:
    """A sparse operator with named domain and range."""

    def __init__(self, tensval=None, row=None, col=None,
                 from_tbl=None, to_tbl=None):
        self.tensval = None if tensval is None else np.asarray(tensval)
        self.row = None if row is None else np.asarray(row, dtype=np.int64)
        self.col = None if col is None else np.asarray(col, dtype=np.int64)
        self.fromTbl = from_tbl
        self.toTbl = to_tbl
        self._matrix: Optional[sps.csr_matrix] = None

    # ----------------------------------------------------------- constructors

    @classmethod
    def from_tensor_map(cls, map_) -> "SparseTensor":
        """The operator a `TensorMap` represents."""
        map_.setup()
        return cls(tensval=np.ones(map_.dispind1.size), row=map_.dispind2,
                   col=map_.dispind1, from_tbl=map_.fromTbl,
                   to_tbl=map_.toTbl)

    @classmethod
    def from_tensor_prod(cls, vec, prod, argindex: int = 1) -> "SparseTensor":
        """Fix argument ``argindex`` of ``prod`` to ``vec``; keep the other free."""
        prod.setup()
        if argindex == 1:
            values = np.asarray(vec)[prod.dispind1]
            return cls(tensval=values, row=prod.dispind3, col=prod.dispind2,
                       from_tbl=prod.tbl2, to_tbl=prod.tbl3)
        if argindex == 2:
            values = np.asarray(vec)[prod.dispind2]
            return cls(tensval=values, row=prod.dispind3, col=prod.dispind1,
                       from_tbl=prod.tbl1, to_tbl=prod.tbl3)
        raise ValueError(f"argindex must be 1 or 2, got {argindex}")

    # -------------------------------------------------------------- as matrix

    def get_matrix(self) -> sps.csr_matrix:
        """The operator as a SciPy sparse matrix; duplicate entries sum."""
        if self._matrix is None:
            values = getattr(self.tensval, "val", self.tensval)
            self._matrix = sps.csr_matrix(
                (np.asarray(values, dtype=float), (self.row, self.col)),
                shape=(self.toTbl.num, self.fromTbl.num))
        return self._matrix

    # ------------------------------------------------------------- operations

    def transpose(self) -> "SparseTensor":
        """Swap domain and range."""
        return SparseTensor(tensval=self.tensval, row=self.col, col=self.row,
                            from_tbl=self.toTbl, to_tbl=self.fromTbl)

    @property
    def T(self) -> "SparseTensor":
        return self.transpose()

    def scale(self, factor: float) -> "SparseTensor":
        """Multiply every coefficient by a scalar."""
        return SparseTensor(tensval=self.tensval * factor, row=self.row,
                            col=self.col, from_tbl=self.fromTbl,
                            to_tbl=self.toTbl)

    def matmul(self, other) -> "SparseTensor":
        """Compose with another operator, or apply to a vector."""
        if isinstance(other, np.ndarray):
            return self.get_matrix() @ other
        if self.fromTbl.num != other.toTbl.num:
            raise ValueError(
                f"cannot compose: the left operator takes {self.fromTbl.num} "
                f"entries, the right produces {other.toTbl.num}")
        product = (self.get_matrix() @ other.get_matrix()).tocoo()
        out = SparseTensor(tensval=product.data, row=product.row,
                           col=product.col, from_tbl=other.fromTbl,
                           to_tbl=self.toTbl)
        out._matrix = product.tocsr()
        return out

    def __matmul__(self, other):
        return self.matmul(other)

    def __mul__(self, factor):
        return self.scale(factor)

    __rmul__ = __mul__

    def plus(self, other) -> "SparseTensor":
        """Add two operators over the same domain and range."""
        for name in ("fromTbl", "toTbl"):
            if getattr(self, name).num != getattr(other, name).num:
                raise ValueError(
                    f"cannot add operators with different {name} sizes: "
                    f"{getattr(self, name).num} and {getattr(other, name).num}")
        total = (self.get_matrix() + other.get_matrix()).tocoo()
        out = SparseTensor(tensval=total.data, row=total.row, col=total.col,
                           from_tbl=self.fromTbl, to_tbl=self.toTbl)
        out._matrix = total.tocsr()
        return out

    def __add__(self, other):
        return self.plus(other)

    def __repr__(self) -> str:
        n = 0 if self.tensval is None else np.size(self.tensval)
        return (f"<SparseTensor {self.toTbl.num}x{self.fromTbl.num}, "
                f"{n} coefficients>")


if __name__ == "__main__":
    from pymrst.utils.tensor_assembly.index_array import IndexArray
    from pymrst.utils.tensor_assembly.tensor_map import TensorMap
    from pymrst.utils.tensor_assembly.tensor_prod import TensorProd

    def cf():
        return IndexArray({"cells": [0, 0, 1], "faces": [3, 7, 7]})

    def cells():
        return IndexArray({"cells": [0, 1]})

    def summing_map():
        return TensorMap(from_tbl=cf(), to_tbl=cells(), mergefds=["cells"])

    def test_a_map_becomes_its_matrix() -> None:
        t = SparseTensor.from_tensor_map(summing_map())
        assert t.get_matrix().shape == (2, 3)
        assert (t.get_matrix() @ np.array([1.0, 10.0, 100.0])).tolist() == \
            [11.0, 100.0]
        print("ok test_a_map_becomes_its_matrix")

    def test_the_matrix_agrees_with_the_map() -> None:
        m = summing_map().setup()
        t = SparseTensor.from_tensor_map(summing_map())
        u = np.array([2.0, 3.0, 5.0])
        assert np.allclose(t.get_matrix() @ u, m.eval(u))
        print("ok test_the_matrix_agrees_with_the_map")

    def test_transpose_swaps_domain_and_range() -> None:
        t = SparseTensor.from_tensor_map(summing_map())
        assert t.T.get_matrix().shape == (3, 2)
        assert np.allclose(t.T.get_matrix().toarray(),
                           t.get_matrix().toarray().T)
        print("ok test_transpose_swaps_domain_and_range")

    def test_fixing_one_argument_of_a_product() -> None:
        """A coefficient field becomes the operator it induces."""
        prod = TensorProd(tbl1=cf(), tbl2=cells(), mergefds=["cells"],
                          reducefds1=["faces"]).setup()
        a = np.array([1.0, 10.0, 100.0])
        t = SparseTensor.from_tensor_prod(a, prod, argindex=1)
        b = np.array([2.0, 3.0])
        assert np.allclose(t.get_matrix() @ b, prod.eval(a, b))
        print("ok test_fixing_one_argument_of_a_product")

    def test_fixing_the_other_argument() -> None:
        prod = TensorProd(tbl1=cf(), tbl2=cells(), mergefds=["cells"],
                          reducefds1=["faces"]).setup()
        a, b = np.array([1.0, 10.0, 100.0]), np.array([2.0, 3.0])
        t = SparseTensor.from_tensor_prod(b, prod, argindex=2)
        assert np.allclose(t.get_matrix() @ a, prod.eval(a, b))
        print("ok test_fixing_the_other_argument")

    def test_duplicate_coordinates_sum() -> None:
        t = SparseTensor(tensval=[1.0, 2.0, 3.0], row=[0, 0, 1], col=[0, 0, 1],
                         from_tbl=cells(), to_tbl=cells())
        assert t.get_matrix().toarray().tolist() == [[3.0, 0.0], [0.0, 3.0]]
        print("ok test_duplicate_coordinates_sum")

    def test_composition_matches_matrix_multiplication() -> None:
        down = SparseTensor.from_tensor_map(summing_map())
        up = SparseTensor.from_tensor_map(
            TensorMap(from_tbl=cells(), to_tbl=cf(), mergefds=["cells"]))
        both = down @ up
        assert both.get_matrix().shape == (2, 2)
        assert np.allclose(both.get_matrix().toarray(),
                           (down.get_matrix() @ up.get_matrix()).toarray())
        print("ok test_composition_matches_matrix_multiplication")

    def test_composition_agrees_with_the_index_array_route() -> None:
        """MRST computes the product through its own join/project machinery
        when no matrix is cached; both must give the same answer."""
        from pymrst.utils.tensor_assembly.cross_index_array import (
            cross_index_array,
        )
        from pymrst.utils.tensor_assembly.proj_index_array import (
            proj_index_array,
        )
        from pymrst.utils.tensor_assembly.utils.tbldispatch1 import tbldispatch1
        from pymrst.utils.tensor_assembly.utils.tbldispatch2 import tbldispatch2

        left = SparseTensor.from_tensor_map(summing_map())
        right = SparseTensor.from_tensor_map(
            TensorMap(from_tbl=cells(), to_tbl=cf(), mergefds=["cells"]))

        t1 = IndexArray({"A": left.row, "B": left.col})
        t2 = IndexArray({"B": right.row, "C": right.col})
        joined, s = cross_index_array(t1, t2, ["B"])
        values = tbldispatch1(left.tensval, s) * tbldispatch2(right.tensval, s)
        out, s3 = proj_index_array(joined, ["A", "C"])
        summed = np.bincount(s3["inds"], weights=values, minlength=out.num)

        direct = (left @ right).get_matrix().toarray()
        for (a, c), v in zip(out.gets(["A", "C"]).tolist(), summed):
            assert np.isclose(direct[a, c], v), (a, c, v, direct[a, c])
        print("ok test_composition_agrees_with_the_index_array_route")

    def test_addition() -> None:
        t = SparseTensor.from_tensor_map(summing_map())
        assert np.allclose((t + t).get_matrix().toarray(),
                           2 * t.get_matrix().toarray())
        print("ok test_addition")

    def test_scaling() -> None:
        t = SparseTensor.from_tensor_map(summing_map())
        assert np.allclose((3.0 * t).get_matrix().toarray(),
                           3.0 * t.get_matrix().toarray())
        print("ok test_scaling")

    def test_applying_to_a_vector() -> None:
        t = SparseTensor.from_tensor_map(summing_map())
        assert (t @ np.array([1.0, 10.0, 100.0])).tolist() == [11.0, 100.0]
        print("ok test_applying_to_a_vector")

    def test_composing_mismatched_operators_is_rejected() -> None:
        t = SparseTensor.from_tensor_map(summing_map())
        try:
            t @ t
        except ValueError as error:
            assert "cannot compose" in str(error)
            print("ok test_composing_mismatched_operators_is_rejected")
            return
        raise AssertionError("a shape mismatch must be rejected")

    def test_adding_mismatched_operators_is_rejected() -> None:
        t = SparseTensor.from_tensor_map(summing_map())
        try:
            t + t.T
        except ValueError as error:
            assert "cannot add" in str(error)
            print("ok test_adding_mismatched_operators_is_rejected")
            return
        raise AssertionError("adding different shapes must be rejected")

    def test_a_bad_argindex_is_rejected() -> None:
        prod = TensorProd(tbl1=cf(), tbl2=cells(), mergefds=["cells"],
                          reducefds1=["faces"]).setup()
        try:
            SparseTensor.from_tensor_prod(np.ones(3), prod, argindex=3)
        except ValueError as error:
            assert "must be 1 or 2" in str(error)
            print("ok test_a_bad_argindex_is_rejected")
            return
        raise AssertionError("an unknown argindex must be rejected")

    test_a_map_becomes_its_matrix()
    test_the_matrix_agrees_with_the_map()
    test_transpose_swaps_domain_and_range()
    test_fixing_one_argument_of_a_product()
    test_fixing_the_other_argument()
    test_duplicate_coordinates_sum()
    test_composition_matches_matrix_multiplication()
    test_composition_agrees_with_the_index_array_route()
    test_addition()
    test_scaling()
    test_applying_to_a_vector()
    test_composing_mismatched_operators_is_rejected()
    test_adding_mismatched_operators_is_rejected()
    test_a_bad_argindex_is_rejected()
    print("\nAll tests passed!")
