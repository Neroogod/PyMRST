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


def _merge_duplicate_entries(data: np.ndarray, row: np.ndarray, col: np.ndarray,
                             ncol: int):
    """Sum entries that share a `(row, col)` instead of keeping both.

    Every caller here concatenates triplets from more than one source
    (`__add__`'s two operands, `matmul_left`'s fallback summing several
    source rows into one output row) and previously left the result as-is:
    two terms landing on the same `(row, col)` - the ordinary case for a
    diagonal Jacobian, where "accumulation + flux + well" all carry a
    d(eq)/d(the cell's own variable) term - became two stored entries
    instead of one summed entry. See `_merge_row_duplicates` in
    `kernels/numba_kernels.py` for the measured cost of this on a real
    compositional model (that one is the row-major twin of this same bug).
    """
    if row.size == 0:
        return data, row, col
    ncol64 = np.int64(ncol)
    key = row.astype(np.int64) * ncol64 + col.astype(np.int64)
    order = np.argsort(key, kind="stable")
    key_sorted = key[order]
    data_sorted = data[order]
    boundaries = np.empty(key_sorted.size, dtype=bool)
    boundaries[0] = True
    boundaries[1:] = key_sorted[1:] != key_sorted[:-1]
    starts = np.flatnonzero(boundaries)
    summed = np.add.reduceat(data_sorted, starts)
    keys_unique = key_sorted[starts]
    row_unique = (keys_unique // ncol64).astype(row.dtype)
    col_unique = (keys_unique % ncol64).astype(col.dtype)
    return summed, row_unique, col_unique


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

    @classmethod
    def vertcat(cls, parts: "list[DiagonalJacobian]") -> "DiagonalJacobian":
        """Stack blocks by row (MRST's ``DiagonalJacobian.vertcat``).

        The fast path MRST's own `cat` reserves for `dim == 1`: every part
        keeps its own row range, columns are untouched, so this is a direct
        triplet append with each part's rows shifted by the rows already
        placed - no sparse round-trip, and the numba kernel (`vstack_triplet`)
        does the shift-and-append in compiled code when it is available.

        All parts must share their column count - MRST asserts the same
        (`dim(2)` must match) since concatenating blocks that differ on which
        variables they differentiate against is a `cat(2, ...)`
        (`horzcat`) question, not this one, and is not fast-pathed here
        either - matching MRST's own `cat`, which falls back to a sparse
        build for `dim == 2`.
        """
        parts = list(parts)
        if not parts:
            return cls.zeros(0, 0)
        ncol = parts[0].shape[1]
        if any(p.shape[1] != ncol for p in parts):
            raise ValueError(
                "vertcat requires every block to share its column count "
                "(the variable set being differentiated against); a "
                "mismatch is MRST's horzcat/cat(2,...) case, not this one")
        data, row, col, nrows = vstack_triplet(
            [p.data for p in parts], [p.row for p in parts],
            [p.col for p in parts], [p.shape[0] for p in parts])
        return cls(data, row, col, (int(nrows), ncol))

    def subset_plus(self, v: "DiagonalJacobian", subs: np.ndarray) -> "DiagonalJacobian":
        """Accumulate ``v`` into the rows at ``subs`` (MRST's ``subsetPlus``).

        ``x.subset_plus(v, subs)`` is MRST's ``x = subsetPlus(x, v, subs)``:
        add ``v`` - a block with ``len(subs)`` rows and this block's column
        count - into rows ``subs`` of ``x``, leaving every other row alone.

        Triplet storage makes this an append, not a read-modify-write: a
        duplicate ``(row, col)`` pair sums when converted to CSR (`to_sparse`
        uses `scipy.sparse.csr_array`, which sums duplicates by construction),
        so remapping `v`'s local rows through `subs` and appending is already
        correct without searching for matching entries first.
        """
        if v.shape[1] != self.shape[1]:
            raise ValueError(
                f"subset_plus needs matching column counts; got "
                f"{self.shape[1]} and {v.shape[1]}")
        subs = np.asarray(subs, dtype=np.int64).ravel()
        if v.shape[0] != subs.size:
            raise ValueError(
                f"subs names {subs.size} rows but v has {v.shape[0]}")
        mapped_row = subs[v.row]
        return DiagonalJacobian(
            np.concatenate([self.data, v.data]),
            np.concatenate([self.row, mapped_row]),
            np.concatenate([self.col, v.col]),
            self.shape)

    subsetPlus = subset_plus

    def subset_minus(self, v: "DiagonalJacobian", subs: np.ndarray) -> "DiagonalJacobian":
        """MRST's ``subsetMinus``: ``subsetPlus(x, -v, subs)``, literally."""
        neg = DiagonalJacobian(-v.data, v.row, v.col, v.shape)
        return self.subset_plus(neg, subs)

    subsetMinus = subset_minus

    def to_sparse(self) -> sp.csr_array:
        return sp.csr_array((self.data, (self.row, self.col)), shape=self.shape)

    def tocsr(self) -> sp.csr_array:
        return self.to_sparse()

    def toarray(self) -> np.ndarray:
        return self.to_sparse().toarray()

    def diagonal(self, k: int = 0) -> np.ndarray:
        return self.to_sparse().diagonal(k)

    # -- MRST DiagonalJacobian.m accessors, ported (Phase 2) ---------------
    #
    # MRST stores an `n x ndiag` *array* plus an optional `subset` and a
    # `rowMajor` flag; this class stores nonzero `(data, row, col)` triplets,
    # for the reason its own docstring gives - TPFA operators produce shifted
    # diagonals that a strict diagonal array cannot hold.
    #
    # So these are ported by *meaning*, not by transcription. Where MRST's
    # method asks a question the triplet store can answer - is this block zero,
    # which rows carry entries, do two blocks share a row pattern - it is
    # implemented. Where the method only makes sense against the array layout
    # (`compareIndices`, `treatSubset`, and `rowMajor` handling generally),
    # transcribing it would invent semantics, so it is left out and recorded in
    # the parity ledger rather than faked.

    def is_zero(self) -> bool:
        """True when the block carries no entries at all.

        Structurally zero, not numerically: a block of explicit zeros is not
        `is_zero`, which matters because the two behave differently under
        assembly - one contributes no sparsity pattern, the other does.
        """
        return self.nnz == 0

    isZero = is_zero

    @staticmethod
    def is_all_zeros(v) -> bool:
        """MRST's `isAllZeros`: a numeric argument that is entirely zero."""
        if isinstance(v, DiagonalJacobian):
            return not np.any(v.data)
        arr = np.asarray(v)
        return arr.dtype.kind in "fiub" and not np.any(arr)

    isAllZeros = is_all_zeros

    def to_zero(self, n: int | None = None) -> "DiagonalJacobian":
        """An empty block, optionally resized to `n` rows."""
        rows = self.shape[0] if n is None else int(n)
        return DiagonalJacobian.zeros(rows, self.shape[1])

    toZero = to_zero

    def expand_zero(self) -> "DiagonalJacobian":
        """Materialise the structural zeros as explicit entries.

        The inverse of the distinction `is_zero` draws: after this the block
        has a full sparsity pattern of zeros, which some assembly paths need
        so the pattern does not change between iterations.
        """
        rows = np.repeat(np.arange(self.shape[0], dtype=np.int64), self.shape[1])
        cols = np.tile(np.arange(self.shape[1], dtype=np.int64), self.shape[0])
        return DiagonalJacobian(np.zeros(rows.size), rows, cols, self.shape)

    expandZero = expand_zero

    def get_subset(self) -> np.ndarray:
        """The rows that carry entries, ascending and without repeats.

        MRST's `subset` is an explicit index list; here it is derived from the
        triplets, which is the same information for the same purpose - knowing
        which rows a block actually touches before combining two of them.
        """
        return np.unique(self.row) if self.nnz else np.zeros(0, dtype=np.int64)

    getSubset = get_subset

    def subsets_equal_no_zero_check(self, other: "DiagonalJacobian") -> bool:
        return np.array_equal(self.get_subset(), other.get_subset())

    subsetsEqualNoZeroCheck = subsets_equal_no_zero_check

    def subsets_equal(self, other: "DiagonalJacobian") -> bool:
        """Whether two blocks touch the same rows.

        An empty block counts as equal to anything, following MRST: it imposes
        no pattern, so combining it with any block is safe. Without that case
        every operation involving a zero block would take the slow path.
        """
        if self.is_zero() or other.is_zero():
            return True
        return self.subsets_equal_no_zero_check(other)

    subsetsEqual = subsets_equal

    def get_number_of_diagonals(self) -> int:
        """How many derivative columns each row carries."""
        return int(self.shape[1])

    getNumberOfDiagonals = get_number_of_diagonals

    def get_diagonal(self, subs=None) -> np.ndarray:
        """The main diagonal, optionally restricted to `subs` rows."""
        values = self.to_sparse().diagonal()
        return values if subs is None else values[subs]

    getDiagonal = get_diagonal

    def set_diagonal(self, diagonals, subs=None) -> "DiagonalJacobian":
        """Write the main diagonal, optionally only at `subs` rows.

        The counterpart of `get_diagonal`, and expressible against the triplet
        store where `compareIndices` and `treatSubset` are not: "the entry at
        (i, i)" means the same thing in both layouts, whereas those two ask
        about MRST's `n x ndiag` column ordering.

        Only *stored* entries are written. Creating one where the sparsity
        pattern has none would change the pattern between iterations, and the
        pattern is what a symbolic factorisation is built on - so a row with no
        diagonal entry is left alone rather than silently gaining one.
        """
        on_diagonal = self.row == self.col
        if subs is not None:
            wanted = np.zeros(self.shape[0], dtype=bool)
            wanted[np.asarray(subs)] = True
            on_diagonal = on_diagonal & wanted[self.row]

        values = np.asarray(diagonals, dtype=float).ravel()
        target = self.row[on_diagonal]
        if values.size == 1:
            self.data[on_diagonal] = values[0]
        elif values.size == target.size:
            self.data[on_diagonal] = values
        else:
            # Indexed by row: take each stored diagonal's own row's value.
            self.data[on_diagonal] = values[target]
        return self

    setDiagonal = set_diagonal

    def get_diagonal_by_index(self, index: int) -> np.ndarray:
        """The main diagonal of derivative block `index`.

        A block spanning several primary variables lays them out as adjacent
        column groups of `nrow` each, so block `index` is columns
        `[index*nrow, (index+1)*nrow)`. Refused when the width is not a whole
        number of blocks, rather than guessing a split.
        """
        nrow, ncol = self.shape
        if nrow == 0 or ncol % nrow:
            raise ValueError(
                f"a {nrow}x{ncol} block is not a whole number of {nrow}-column "
                f"derivative blocks; cannot index it by variable")
        count = ncol//nrow
        if not 0 <= index < count:
            raise IndexError(f"block {index} of {count}")
        start = index*nrow
        mask = (self.col >= start) & (self.col < start + nrow)
        out = np.zeros(nrow)
        rows, cols = self.row[mask], self.col[mask] - start
        keep = rows == cols
        out[rows[keep]] = self.data[mask][keep]
        return out

    getDiagonalByIndex = get_diagonal_by_index

    def repmat(self, row_reps: int, col_reps: int = 1):
        """Tile the block. Returns sparse, as MRST does - a tiled diagonal is
        not itself diagonal."""
        return sp.csr_array(sp.bmat([[self.to_sparse()]*col_reps]*row_reps))

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
        data, row, col = _merge_duplicate_entries(
            np.concatenate(pieces_data),
            np.concatenate(pieces_row),
            np.concatenate(pieces_col),
            self.shape[1],
        )
        return DiagonalJacobian(data, row, col, (mat.shape[0], self.shape[1]))

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
        data, row, col = _merge_duplicate_entries(
            np.concatenate([self.data, other.data]),
            np.concatenate([self.row, other.row]),
            np.concatenate([self.col, other.col]),
            self.shape[1],
        )
        return DiagonalJacobian(data, row, col, self.shape)

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

    def test_matmul_left_merges_when_two_source_rows_share_a_column():
        """`mat`'s row 0 combines `j`'s src_row 0 and src_row 2, which both
        carry a column-0 entry; row 1 combines src_row 1 and src_row 2,
        which both carry a column-2 entry. `matmul_left` must merge each
        into one stored entry, the same class of bug `__add__`'s test
        below pins. `ref.nnz` is the ground truth: scipy's own CSR@CSR
        product never leaves explicit duplicate indices, so it is the
        true unique-entry count without hand-computing one."""
        j = DiagonalJacobian.from_sparse(sp.csr_array([[1.0, 2.0, 0.0],
                                                       [0.0, 3.0, 4.0],
                                                       [5.0, 0.0, 6.0]]))
        mat = sp.csr_array([[2.0, 0.0, -1.0],
                            [0.0, 0.5, 1.0]])
        out = j.matmul_left(mat)
        ref = mat @ j.to_sparse()
        assert np.allclose(out.toarray(), ref.toarray())
        assert out.nnz == ref.nnz, (out.nnz, ref.nnz)
        print("ok test_matmul_left_merges_when_two_source_rows_share_a_column")

    def test_direct_addition_merges_shared_row_col_entries():
        """`a` and `b` both carry a (0, 0) entry; `__add__` must sum them
        into one stored entry, not keep both. Pins the fix for the bug
        `_merge_duplicate_entries`/`_merge_row_duplicates` close: this used
        to assert `out.nnz == a.nnz + b.nnz` (4), pinning unmerged
        concatenation as intended behavior. The true unique-entry count is
        3 - (0,0) merged, (1,0) and (1,1) each appear in only one operand."""
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
        assert out.nnz == 3, out.nnz
        print("ok test_direct_addition_merges_shared_row_col_entries")

    def test_vertcat_matches_sparse_vstack():
        """MRST's `vertcat` fast path, checked against the definition it
        replaces: stacking must equal converting each part to sparse,
        `scipy.sparse.vstack`-ing, and converting back."""
        a = DiagonalJacobian.from_sparse(sp.csr_array(
            [[1.0, 2.0, 0.0], [0.0, 3.0, 4.0]]))
        b = DiagonalJacobian.from_sparse(sp.csr_array([[5.0, 0.0, 6.0]]))
        c = DiagonalJacobian.zeros(2, 3)
        out = DiagonalJacobian.vertcat([a, b, c])
        ref = sp.vstack([a.to_sparse(), b.to_sparse(), c.to_sparse()])
        assert out.shape == (5, 3)
        assert np.allclose(out.toarray(), ref.toarray())
        print("ok test_vertcat_matches_sparse_vstack")

    def test_vertcat_rejects_mismatched_column_counts():
        a = DiagonalJacobian.identity(2)
        b = DiagonalJacobian.identity(3)
        try:
            DiagonalJacobian.vertcat([a, b])
        except ValueError as error:
            assert "column count" in str(error)
            print("ok test_vertcat_rejects_mismatched_column_counts")
            return
        raise AssertionError("blocks with different column counts must not "
                             "silently concatenate")

    def test_subset_plus_matches_manual_row_add():
        x = DiagonalJacobian.from_sparse(sp.csr_array(
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]))
        v = DiagonalJacobian.from_sparse(sp.csr_array([[0.0, 9.0]]))
        out = x.subset_plus(v, np.array([1]))
        expect = x.toarray().copy()
        expect[1, :] += v.toarray()[0, :]
        assert np.allclose(out.toarray(), expect)
        # every OTHER row must be untouched
        assert np.allclose(out.toarray()[0], x.toarray()[0])
        assert np.allclose(out.toarray()[2], x.toarray()[2])
        print("ok test_subset_plus_matches_manual_row_add")

    def test_subset_minus_is_subset_plus_of_the_negation():
        x = DiagonalJacobian.identity(3)
        v = DiagonalJacobian.from_sparse(sp.csr_array([[4.0, 0.0, 0.0]]))
        out = x.subset_minus(v, np.array([0]))
        assert np.isclose(out.toarray()[0, 0], 1.0 - 4.0)
        print("ok test_subset_minus_is_subset_plus_of_the_negation")

    def test_subset_plus_accumulates_repeated_targets():
        """Two contributions landing on the SAME row must both count - this
        is what makes it an accumulate rather than a last-write-wins set,
        and is the case a naive dict-keyed-by-row implementation gets wrong."""
        x = DiagonalJacobian.zeros(2, 1)
        v1 = DiagonalJacobian(np.array([3.0]), np.array([0]), np.array([0]), (1, 1))
        v2 = DiagonalJacobian(np.array([5.0]), np.array([0]), np.array([0]), (1, 1))
        out = x.subset_plus(v1, np.array([0])).subset_plus(v2, np.array([0]))
        assert np.isclose(out.toarray()[0, 0], 8.0)
        print("ok test_subset_plus_accumulates_repeated_targets")

    test_identity_and_sparse_roundtrip()
    test_row_selection_and_scaling()
    test_direct_matmul_left_matches_sparse_product()
    test_matmul_left_merges_when_two_source_rows_share_a_column()
    test_direct_addition_merges_shared_row_col_entries()
    test_vertcat_matches_sparse_vstack()
    test_vertcat_rejects_mismatched_column_counts()
    test_subset_plus_matches_manual_row_add()
    test_subset_minus_is_subset_plus_of_the_negation()
    test_subset_plus_accumulates_repeated_targets()
    print("\nAll tests passed!")
