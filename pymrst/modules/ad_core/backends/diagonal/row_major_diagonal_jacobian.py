"""Row-major diagonal-backend Jacobian storage.

MRST's fastest diagonal backend stores derivative data in a row-major layout for
the MEX kernels. PyMRST ports the layout directly and uses optional Numba kernels
for the row-local operations when they are installed. Entries are grouped by
equation row through CSR-style ``indptr``/``col`` arrays, with explicit sparse
conversion at the linear solve.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.backends.diagonal.kernels import (
    HAS_NUMBA_KERNELS,
    add_row_major,
    left_scale_row_major,
    matmul_left_row_major,
    row_repeat_row_major,
    take_rows_row_major,
    vstack_row_major,
)


@dataclass
class RowMajorDiagonalJacobian:
    """Compact row-major Jacobian block."""

    data: np.ndarray
    col: np.ndarray
    indptr: np.ndarray
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64).ravel()
        self.col = np.asarray(self.col, dtype=np.int64).ravel()
        self.indptr = np.asarray(self.indptr, dtype=np.int64).ravel()
        self.shape = (int(self.shape[0]), int(self.shape[1]))
        if self.data.size != self.col.size:
            raise ValueError("data and col must have the same length")
        if self.indptr.size != self.shape[0] + 1:
            raise ValueError("indptr length must be nrow + 1")
        if self.indptr[0] != 0 or self.indptr[-1] != self.data.size:
            raise ValueError("indptr endpoints do not match data length")

    @classmethod
    def zeros(cls, nrow: int, ncol: int) -> "RowMajorDiagonalJacobian":
        return cls(np.array([], dtype=float), np.array([], dtype=np.int64),
                   np.zeros(int(nrow) + 1, dtype=np.int64), (nrow, ncol))

    @classmethod
    def identity(cls, n: int) -> "RowMajorDiagonalJacobian":
        ind = np.arange(n, dtype=np.int64)
        return cls(np.ones(n), ind, np.arange(n + 1, dtype=np.int64), (n, n))

    @classmethod
    def from_sparse(cls, mat) -> "RowMajorDiagonalJacobian":
        csr = sp.csr_array(mat)
        return cls(csr.data.copy(), csr.indices.copy(), csr.indptr.copy(), csr.shape)

    @property
    def nnz(self) -> int:
        return int(self.data.size)

    def copy(self) -> "RowMajorDiagonalJacobian":
        return RowMajorDiagonalJacobian(self.data.copy(), self.col.copy(),
                                        self.indptr.copy(), self.shape)

    @classmethod
    def vertcat(cls, parts: "list[RowMajorDiagonalJacobian]") -> "RowMajorDiagonalJacobian":
        """Stack blocks by row - MRST's `DiagonalJacobian.vertcat`, on the
        row-major layout. See `DiagonalJacobian.vertcat`, whose docstring
        this mirrors: only `dim == 1` is fast-pathed, matching MRST's own
        `cat`, and every part must share its column count."""
        parts = list(parts)
        if not parts:
            return cls.zeros(0, 0)
        ncol = parts[0].shape[1]
        if any(p.shape[1] != ncol for p in parts):
            raise ValueError(
                "vertcat requires every block to share its column count")
        data, col, indptr, nrows = vstack_row_major(
            [p.data for p in parts], [p.col for p in parts],
            [p.indptr for p in parts], [p.shape[0] for p in parts])
        return cls(data, col, indptr, (int(nrows), ncol))

    def subset_plus(self, v: "RowMajorDiagonalJacobian",
                    subs: np.ndarray) -> "RowMajorDiagonalJacobian":
        """MRST's `subsetPlus`: add `v` into the rows at `subs`.

        Composed from `to_sparse`/`from_sparse` rather than a dedicated
        kernel - unlike `vertcat`, this is not on `vertcat_adi`'s hot path,
        and a correct sparse composition is preferable to a new, less-tested
        native code path for an operation nothing here currently calls under
        load.
        """
        if v.shape[1] != self.shape[1]:
            raise ValueError(
                f"subset_plus needs matching column counts; got "
                f"{self.shape[1]} and {v.shape[1]}")
        subs = np.asarray(subs, dtype=np.int64).ravel()
        if v.shape[0] != subs.size:
            raise ValueError(
                f"subs names {subs.size} rows but v has {v.shape[0]}")
        out = self.to_sparse().tolil()
        out[subs, :] = out[subs, :] + v.to_sparse()
        return RowMajorDiagonalJacobian.from_sparse(out.tocsr())

    subsetPlus = subset_plus

    def subset_minus(self, v: "RowMajorDiagonalJacobian",
                     subs: np.ndarray) -> "RowMajorDiagonalJacobian":
        """MRST's `subsetMinus`: `subsetPlus(x, -v, subs)`, literally."""
        neg = RowMajorDiagonalJacobian(-v.data, v.col, v.indptr, v.shape)
        return self.subset_plus(neg, subs)

    subsetMinus = subset_minus

    def to_sparse(self) -> sp.csr_array:
        return sp.csr_array((self.data, self.col, self.indptr), shape=self.shape)

    def tocsr(self) -> sp.csr_array:
        return self.to_sparse()

    def toarray(self) -> np.ndarray:
        return self.to_sparse().toarray()

    def diagonal(self, k: int = 0) -> np.ndarray:
        return self.to_sparse().diagonal(k)

    def left_scale(self, scale) -> "RowMajorDiagonalJacobian":
        scale = np.asarray(scale, dtype=np.float64).ravel()
        if scale.size == 1:
            return RowMajorDiagonalJacobian(self.data * scale[0], self.col.copy(),
                                            self.indptr.copy(), self.shape)
        if scale.size != self.shape[0]:
            raise ValueError(f"scale length {scale.size} does not match rows {self.shape[0]}")
        if HAS_NUMBA_KERNELS and left_scale_row_major is not None:
            return RowMajorDiagonalJacobian(
                left_scale_row_major(self.data, self.indptr, scale),
                self.col.copy(),
                self.indptr.copy(),
                self.shape,
            )
        row_counts = np.diff(self.indptr)
        return RowMajorDiagonalJacobian(
            self.data * np.repeat(scale, row_counts),
            self.col.copy(),
            self.indptr.copy(),
            self.shape,
        )

    def matmul_left(self, mat) -> "RowMajorDiagonalJacobian":
        mat = sp.csr_array(mat)
        if mat.shape[1] != self.shape[0]:
            raise ValueError(f"matrix shape {mat.shape} cannot multiply Jacobian shape {self.shape}")
        if self.nnz == 0 or mat.nnz == 0:
            return RowMajorDiagonalJacobian.zeros(mat.shape[0], self.shape[1])
        if HAS_NUMBA_KERNELS and matmul_left_row_major is not None:
            data, col, indptr = matmul_left_row_major(
                mat.data, mat.indices, mat.indptr, self.data, self.col, self.indptr
            )
            return RowMajorDiagonalJacobian(data, col, indptr, (mat.shape[0], self.shape[1]))

        chunks_data = []
        chunks_col = []
        indptr = np.zeros(mat.shape[0] + 1, dtype=np.int64)
        cursor = 0
        for out_row in range(mat.shape[0]):
            for aptr in range(mat.indptr[out_row], mat.indptr[out_row + 1]):
                src_row = mat.indices[aptr]
                factor = mat.data[aptr]
                start, stop = self.indptr[src_row], self.indptr[src_row + 1]
                if stop > start:
                    chunks_data.append(factor * self.data[start:stop])
                    chunks_col.append(self.col[start:stop])
                    cursor += stop - start
            indptr[out_row + 1] = cursor
        data = np.concatenate(chunks_data) if chunks_data else np.array([], dtype=float)
        col = np.concatenate(chunks_col) if chunks_col else np.array([], dtype=np.int64)
        return RowMajorDiagonalJacobian(data, col, indptr, (mat.shape[0], self.shape[1]))

    def take_rows(self, idx) -> "RowMajorDiagonalJacobian":
        rows = np.arange(self.shape[0], dtype=np.int64)[idx]
        rows = np.atleast_1d(rows)
        if HAS_NUMBA_KERNELS and take_rows_row_major is not None:
            data, col, indptr = take_rows_row_major(self.data, self.col, self.indptr, rows)
            return RowMajorDiagonalJacobian(data, col, indptr, (rows.size, self.shape[1]))
        chunks_data = []
        chunks_col = []
        indptr = np.zeros(rows.size + 1, dtype=np.int64)
        nnz = 0
        for out_row, row in enumerate(rows):
            start, stop = self.indptr[row], self.indptr[row + 1]
            chunks_data.append(self.data[start:stop])
            chunks_col.append(self.col[start:stop])
            nnz += stop - start
            indptr[out_row + 1] = nnz
        data = np.concatenate(chunks_data) if chunks_data else np.array([], dtype=float)
        col = np.concatenate(chunks_col) if chunks_col else np.array([], dtype=np.int64)
        return RowMajorDiagonalJacobian(data, col, indptr, (rows.size, self.shape[1]))

    def row_repeat(self, n: int) -> "RowMajorDiagonalJacobian":
        if self.shape[0] != 1:
            raise ValueError("only one-row Jacobians can be row-repeated")
        if self.nnz == 0:
            return RowMajorDiagonalJacobian.zeros(n, self.shape[1])
        if HAS_NUMBA_KERNELS and row_repeat_row_major is not None:
            data, col, indptr = row_repeat_row_major(self.data, self.col, n)
            return RowMajorDiagonalJacobian(data, col, indptr, (n, self.shape[1]))
        return RowMajorDiagonalJacobian(
            np.tile(self.data, n),
            np.tile(self.col, n),
            np.arange(n + 1, dtype=np.int64) * self.nnz,
            (n, self.shape[1]),
        )

    def __neg__(self) -> "RowMajorDiagonalJacobian":
        return RowMajorDiagonalJacobian(-self.data, self.col.copy(),
                                        self.indptr.copy(), self.shape)

    def __add__(self, other) -> "RowMajorDiagonalJacobian":
        if self.shape != other.shape:
            raise ValueError(f"Jacobian shapes differ: {self.shape} vs {other.shape}")
        if HAS_NUMBA_KERNELS and add_row_major is not None:
            data, col, indptr = add_row_major(
                self.data, self.col, self.indptr,
                other.data, other.col, other.indptr,
            )
            return RowMajorDiagonalJacobian(data, col, indptr, self.shape)
        nrow = self.shape[0]
        nnz = self.nnz + other.nnz
        data = np.empty(nnz, dtype=np.float64)
        col = np.empty(nnz, dtype=np.int64)
        indptr = np.empty(nrow + 1, dtype=np.int64)
        indptr[0] = 0
        cursor = 0
        for row in range(nrow):
            for block in (self, other):
                start, stop = block.indptr[row], block.indptr[row + 1]
                count = stop - start
                if count:
                    data[cursor:cursor + count] = block.data[start:stop]
                    col[cursor:cursor + count] = block.col[start:stop]
                    cursor += count
            indptr[row + 1] = cursor
        return RowMajorDiagonalJacobian(data, col, indptr, self.shape)

    def __sub__(self, other) -> "RowMajorDiagonalJacobian":
        return self + (-other)

    def __mul__(self, scalar: float) -> "RowMajorDiagonalJacobian":
        return RowMajorDiagonalJacobian(self.data * float(scalar), self.col.copy(),
                                        self.indptr.copy(), self.shape)

    __rmul__ = __mul__


if __name__ == "__main__":
    def test_row_major_identity_and_sparse_roundtrip():
        j = RowMajorDiagonalJacobian.identity(3)
        assert np.allclose(j.toarray(), np.eye(3))
        assert RowMajorDiagonalJacobian.from_sparse(j.to_sparse()).nnz == 3
        print("ok test_row_major_identity_and_sparse_roundtrip")

    def test_row_major_row_ops():
        j = RowMajorDiagonalJacobian.identity(4)
        picked = j.take_rows([0, 3]).left_scale([2.0, -1.0])
        assert np.allclose(picked.toarray(), [[2, 0, 0, 0], [0, 0, 0, -1]])
        print("ok test_row_major_row_ops")

    def test_row_major_direct_matmul_left_matches_sparse_product():
        """`mat`'s row 0 combines `j`'s src_row 0 and src_row 2 (both carry
        a column-0 entry); row 1 combines src_row 1 and src_row 2 (both
        carry a column-2 entry). `matmul_left` must merge each pair into
        one stored entry - `ref.nnz` is the ground truth, since scipy's
        own CSR@CSR product never leaves explicit duplicate indices."""
        j = RowMajorDiagonalJacobian.from_sparse(sp.csr_array([[1.0, 2.0, 0.0],
                                                               [0.0, 3.0, 4.0],
                                                               [5.0, 0.0, 6.0]]))
        mat = sp.csr_array([[2.0, 0.0, -1.0],
                            [0.0, 0.5, 1.0]])
        out = j.matmul_left(mat)
        ref = mat @ j.to_sparse()
        assert np.allclose(out.toarray(), ref.toarray())
        assert out.nnz == ref.nnz, (out.nnz, ref.nnz)
        print("ok test_row_major_direct_matmul_left_matches_sparse_product")

    def test_row_major_direct_addition_merges_shared_row_col_entries():
        """`a` and `b` both carry a row-0/col-0 entry; `__add__` must sum
        them into one stored entry, not keep both. Pins the fix for the
        bug `_merge_row_duplicates` closes: this used to assert
        `out.nnz == a.nnz + b.nnz` (4), pinning unmerged concatenation as
        intended behavior. True unique count is 3 - row0 col0 merged
        (1+3=4), row1 keeps its col0 (-4) and col1 (2) separately."""
        a = RowMajorDiagonalJacobian.from_sparse(sp.csr_array([[1.0, 0.0],
                                                               [0.0, 2.0]]))
        b = RowMajorDiagonalJacobian.from_sparse(sp.csr_array([[3.0, 0.0],
                                                               [-4.0, 0.0]]))
        out = a + b
        ref = a.to_sparse() + b.to_sparse()
        assert np.allclose(out.toarray(), ref.toarray())
        assert out.nnz == 3, out.nnz
        print("ok test_row_major_direct_addition_merges_shared_row_col_entries")

    def test_row_major_vertcat_matches_sparse_vstack():
        a = RowMajorDiagonalJacobian.from_sparse(sp.csr_array(
            [[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]]))
        b = RowMajorDiagonalJacobian.from_sparse(sp.csr_array([[5.0, 0.0, 6.0]]))
        out = RowMajorDiagonalJacobian.vertcat([a, b])
        ref = sp.vstack([a.to_sparse(), b.to_sparse()])
        assert out.shape == (3, 3)
        assert np.allclose(out.toarray(), ref.toarray())
        print("ok test_row_major_vertcat_matches_sparse_vstack")

    def test_row_major_subset_plus_matches_manual_row_add():
        x = RowMajorDiagonalJacobian.from_sparse(sp.csr_array(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]))
        v = RowMajorDiagonalJacobian.from_sparse(sp.csr_array([[0.0, 9.0]]))
        out = x.subset_plus(v, np.array([1]))
        expect = x.toarray().copy()
        expect[1, :] += v.toarray()[0, :]
        assert np.allclose(out.toarray(), expect)
        assert np.allclose(out.toarray()[0], x.toarray()[0])
        assert np.allclose(out.toarray()[2], x.toarray()[2])
        print("ok test_row_major_subset_plus_matches_manual_row_add")

    def test_row_major_subset_minus_is_subset_plus_of_the_negation():
        x = RowMajorDiagonalJacobian.identity(2)
        v = RowMajorDiagonalJacobian.from_sparse(sp.csr_array([[4.0, 0.0]]))
        out = x.subset_minus(v, np.array([0]))
        assert np.isclose(out.toarray()[0, 0], 1.0 - 4.0)
        print("ok test_row_major_subset_minus_is_subset_plus_of_the_negation")

    test_row_major_identity_and_sparse_roundtrip()
    test_row_major_row_ops()
    test_row_major_direct_matmul_left_matches_sparse_product()
    test_row_major_direct_addition_merges_shared_row_col_entries()
    test_row_major_vertcat_matches_sparse_vstack()
    test_row_major_subset_plus_matches_manual_row_add()
    test_row_major_subset_minus_is_subset_plus_of_the_negation()
    print("\nAll tests passed!")
