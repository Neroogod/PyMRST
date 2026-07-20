"""Diagonal-style Jacobian storage used by PyMRST's diagonal AD backend.

MRST's ``DiagonalJacobian`` stores local derivative arrays compactly and only
assembles sparse matrices when the nonlinear solver asks for a linear system.
This Python port follows the same contract: AD operators work on compact row,
column, value triplets, and ``to_sparse`` is the explicit sparse handoff.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from pymrst.modules.ad_core.backends.diagonal.kernels import (
    HAS_NUMBA_KERNELS,
    matmul_left_triplet,
    row_repeat_triplet,
    take_rows_triplet,
    vstack_triplet,
)


@dataclass
class DiagonalJacobian:
    """Compact Jacobian block with explicit sparse conversion.

    The name follows MRST. For cell-wise algebra the storage is strictly
    diagonal. TPFA operators introduce shifted diagonals/stencils, so we retain
    the compact nonzero triplets rather than eagerly materializing CSR blocks.
    """

    data: np.ndarray
    row: np.ndarray
    col: np.ndarray
    shape: tuple[int, int]

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64).ravel()
        self.row = np.asarray(self.row, dtype=np.int64).ravel()
        self.col = np.asarray(self.col, dtype=np.int64).ravel()
        if not (self.data.size == self.row.size == self.col.size):
            raise ValueError("data, row, and col must have the same length")
        self.shape = (int(self.shape[0]), int(self.shape[1]))

    @classmethod
    def zeros(cls, nrow: int, ncol: int) -> "DiagonalJacobian":
        return cls(np.array([], dtype=float), np.array([], dtype=np.int64),
                   np.array([], dtype=np.int64), (nrow, ncol))

    @classmethod
    def identity(cls, n: int) -> "DiagonalJacobian":
        ind = np.arange(n, dtype=np.int64)
        return cls(np.ones(n), ind, ind, (n, n))

    @classmethod
    def from_sparse(cls, mat) -> "DiagonalJacobian":
        coo = sp.coo_array(mat)
        return cls(coo.data.copy(), coo.row.copy(), coo.col.copy(), coo.shape)

    @property
    def nnz(self) -> int:
        return int(self.data.size)

    def copy(self) -> "DiagonalJacobian":
        return DiagonalJacobian(self.data.copy(), self.row.copy(), self.col.copy(), self.shape)

    def to_sparse(self) -> sp.csr_array:
        return sp.csr_array((self.data, (self.row, self.col)), shape=self.shape)

    def tocsr(self) -> sp.csr_array:
        return self.to_sparse()

    def toarray(self) -> np.ndarray:
        return self.to_sparse().toarray()

    def diagonal(self, k: int = 0) -> np.ndarray:
        return self.to_sparse().diagonal(k)

    def left_scale(self, scale) -> "DiagonalJacobian":
        scale = np.asarray(scale, dtype=np.float64).ravel()
        if scale.size == 1:
            return DiagonalJacobian(self.data * scale[0], self.row.copy(), self.col.copy(), self.shape)
        if scale.size != self.shape[0]:
            raise ValueError(f"scale length {scale.size} does not match rows {self.shape[0]}")
        return DiagonalJacobian(self.data * scale[self.row], self.row.copy(), self.col.copy(), self.shape)

    def matmul_left(self, mat) -> "DiagonalJacobian":
        mat = sp.csr_array(mat)
        if mat.shape[1] != self.shape[0]:
            raise ValueError(f"matrix shape {mat.shape} cannot multiply Jacobian shape {self.shape}")
        if HAS_NUMBA_KERNELS and matmul_left_triplet is not None:
            data, row, col = matmul_left_triplet(
                mat.data, mat.indices, mat.indptr, self.data, self.row, self.col
            )
            return DiagonalJacobian(data, row, col, (mat.shape[0], self.shape[1]))
        coo = mat.tocoo()
        if coo.nnz == 0 or self.nnz == 0:
            return DiagonalJacobian.zeros(mat.shape[0], self.shape[1])
        pieces_data, pieces_row, pieces_col = [], [], []
        for factor, out_row, src_row in zip(coo.data, coo.row, coo.col):
            mask = self.row == src_row
            if np.any(mask):
                pieces_data.append(factor * self.data[mask])
                pieces_row.append(np.full(np.count_nonzero(mask), out_row, dtype=np.int64))
                pieces_col.append(self.col[mask])
        if not pieces_data:
            return DiagonalJacobian.zeros(mat.shape[0], self.shape[1])
        return DiagonalJacobian(
            np.concatenate(pieces_data),
            np.concatenate(pieces_row),
            np.concatenate(pieces_col),
            (mat.shape[0], self.shape[1]),
        )

    def take_rows(self, idx) -> "DiagonalJacobian":
        rows = np.arange(self.shape[0], dtype=np.int64)[idx]
        rows = np.atleast_1d(rows)
        if HAS_NUMBA_KERNELS and take_rows_triplet is not None:
            data, row, col = take_rows_triplet(self.data, self.row, self.col, rows)
            return DiagonalJacobian(data, row, col, (rows.size, self.shape[1]))
        selector = sp.csr_array(
            (np.ones(rows.size), (np.arange(rows.size), rows)),
            shape=(rows.size, self.shape[0]),
        )
        return self.matmul_left(selector)

    def row_repeat(self, n: int) -> "DiagonalJacobian":
        if self.shape[0] != 1:
            raise ValueError("only one-row Jacobians can be row-repeated")
        if self.nnz == 0:
            return DiagonalJacobian.zeros(n, self.shape[1])
        if HAS_NUMBA_KERNELS and row_repeat_triplet is not None:
            data, row, col = row_repeat_triplet(self.data, self.col, n)
            return DiagonalJacobian(data, row, col, (n, self.shape[1]))
        row = np.repeat(np.arange(n, dtype=np.int64), self.nnz)
        col = np.tile(self.col, n)
        data = np.tile(self.data, n)
        return DiagonalJacobian(data, row, col, (n, self.shape[1]))

    def __neg__(self) -> "DiagonalJacobian":
        return DiagonalJacobian(-self.data, self.row.copy(), self.col.copy(), self.shape)

    def __add__(self, other: "DiagonalJacobian") -> "DiagonalJacobian":
        if self.shape != other.shape:
            raise ValueError(f"Jacobian shapes differ: {self.shape} vs {other.shape}")
        return DiagonalJacobian(
            np.concatenate([self.data, other.data]),
            np.concatenate([self.row, other.row]),
            np.concatenate([self.col, other.col]),
            self.shape,
        )

    def __sub__(self, other: "DiagonalJacobian") -> "DiagonalJacobian":
        return self + (-other)

    def __mul__(self, scalar: float) -> "DiagonalJacobian":
        scalar = float(scalar)
        return DiagonalJacobian(self.data * scalar, self.row.copy(), self.col.copy(), self.shape)

    __rmul__ = __mul__


def diagonal_hstack(blocks: list[DiagonalJacobian]) -> sp.csr_array:
    """Assemble block columns for the linear solve handoff."""
    return sp.hstack([block.to_sparse() for block in blocks], format="csr")


def diagonal_vstack(blocks: list[DiagonalJacobian]) -> list[DiagonalJacobian]:
    """Stack matching Jacobian blocks vertically without losing compact storage."""
    if not blocks:
        return []
    ncol = blocks[0].shape[1]
    data, row, col = [], [], []
    offset = 0
    for block in blocks:
        if block.shape[1] != ncol:
            raise ValueError("all Jacobian blocks must have the same column count")
    if HAS_NUMBA_KERNELS and vstack_triplet is not None:
        data, row, col, nrow = vstack_triplet(
            [block.data for block in blocks],
            [block.row for block in blocks],
            [block.col for block in blocks],
            [block.shape[0] for block in blocks],
        )
        return [DiagonalJacobian(data, row, col, (nrow, ncol))]
    for block in blocks:
        data.append(block.data)
        row.append(block.row + offset)
        col.append(block.col)
        offset += block.shape[0]
    return [DiagonalJacobian(np.concatenate(data), np.concatenate(row), np.concatenate(col), (offset, ncol))]


if __name__ == "__main__":
    def test_identity_and_sparse_roundtrip():
        j = DiagonalJacobian.identity(3)
        assert np.allclose(j.toarray(), np.eye(3))
        assert DiagonalJacobian.from_sparse(j.to_sparse()).nnz == 3
        print("ok test_identity_and_sparse_roundtrip")

    def test_row_selection_and_scaling():
        j = DiagonalJacobian.identity(4)
        picked = j.take_rows([0, 3]).left_scale([2.0, -1.0])
        assert np.allclose(picked.toarray(), [[2, 0, 0, 0], [0, 0, 0, -1]])
        print("ok test_row_selection_and_scaling")

    def test_direct_matmul_left_matches_sparse_product():
        j = DiagonalJacobian.from_sparse(sp.csr_array([[1.0, 2.0, 0.0],
                                                       [0.0, 3.0, 4.0],
                                                       [5.0, 0.0, 6.0]]))
        mat = sp.csr_array([[2.0, 0.0, -1.0],
                            [0.0, 0.5, 1.0]])
        out = j.matmul_left(mat)
        ref = mat @ j.to_sparse()
        assert np.allclose(out.toarray(), ref.toarray())
        print("ok test_direct_matmul_left_matches_sparse_product")

    def test_direct_addition_matches_sparse_sum_with_duplicate_entries():
        a = DiagonalJacobian(
            np.array([1.0, 2.0]),
            np.array([0, 1]),
            np.array([0, 1]),
            (2, 2),
        )
        b = DiagonalJacobian(
            np.array([3.0, -4.0]),
            np.array([0, 1]),
            np.array([0, 0]),
            (2, 2),
        )
        out = a + b
        ref = a.to_sparse() + b.to_sparse()
        assert np.allclose(out.toarray(), ref.toarray())
        assert out.nnz == a.nnz + b.nnz
        print("ok test_direct_addition_matches_sparse_sum_with_duplicate_entries")

    test_identity_and_sparse_roundtrip()
    test_row_selection_and_scaling()
    test_direct_matmul_left_matches_sparse_product()
    test_direct_addition_matches_sparse_sum_with_duplicate_entries()
    print("\nAll tests passed!")
