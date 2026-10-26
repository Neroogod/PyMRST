"""Numba kernels for MRST-style row-major diagonal Jacobians.

These kernels are the PyMRST analogue of MRST's compiled diagonal/MEX helpers:
they accelerate the compact row-major storage operations while keeping the same
public Python objects and sparse conversion path for the linear solve.
"""

from __future__ import annotations

import numpy as np
from numba import njit

HAS_NUMBA_KERNELS = True


@njit(cache=True)
def _merge_row_duplicates(out_data, out_col, row_start, row_end):
    """Compact ``out_data``/``out_col[row_start:row_end]`` in place, summing
    entries that share a column instead of keeping both.

    Every kernel below that assembles one output row from more than one
    source (adding two blocks, or left-multiplying by a matrix whose row
    references more than one source row) previously just concatenated the
    pieces: two terms landing on the same `(row, col)` - the ordinary case
    for a diagonal Jacobian, where "accumulation + flux + well" all carry a
    d(eq)/d(the cell's own variable) term - became two stored entries
    instead of one summed entry. `nnz` then grew with the number of terms
    ever added, not the number of actual dependencies, and every later
    operation on that block (scale, matmul, the final CSR conversion for
    the linear solve) paid for the inflated count. Measured on a
    compositional model step: `add_row_major` alone was called 23,184
    times and cost 9.93s of a 56.4s step, and a 5-term chain of 2000-row
    identity blocks bloated from 2000 to 12000 nnz (6x) with the true
    unique-entry count never exceeding 2000.

    Linear per-row scan, not a sort: a real Jacobian row has a handful of
    dependencies (self, a few neighbors, a few phases), not hundreds, so
    the O(k^2) worst case here is cheap in practice and avoids depending on
    `np.argsort`'s stability guarantees inside `nopython` mode.
    """
    write = row_start
    for read in range(row_start, row_end):
        c = out_col[read]
        d = out_data[read]
        matched = -1
        for k in range(row_start, write):
            if out_col[k] == c:
                matched = k
                break
        if matched >= 0:
            out_data[matched] += d
        else:
            out_col[write] = c
            out_data[write] = d
            write += 1
    return write


@njit(cache=True)
def _left_scale_row_major(data, indptr, scale):
    out = np.empty_like(data)
    for row in range(indptr.size - 1):
        factor = scale[row]
        for ptr in range(indptr[row], indptr[row + 1]):
            out[ptr] = data[ptr] * factor
    return out


@njit(cache=True)
def _take_rows_row_major(data, col, indptr, rows):
    nnz = 0
    for i in range(rows.size):
        row = rows[i]
        nnz += indptr[row + 1] - indptr[row]

    out_data = np.empty(nnz, dtype=data.dtype)
    out_col = np.empty(nnz, dtype=col.dtype)
    out_indptr = np.empty(rows.size + 1, dtype=indptr.dtype)
    out_indptr[0] = 0

    cursor = 0
    for out_row in range(rows.size):
        row = rows[out_row]
        start = indptr[row]
        stop = indptr[row + 1]
        for ptr in range(start, stop):
            out_data[cursor] = data[ptr]
            out_col[cursor] = col[ptr]
            cursor += 1
        out_indptr[out_row + 1] = cursor
    return out_data, out_col, out_indptr


@njit(cache=True)
def _row_repeat_row_major(data, col, n):
    nnz_per_row = data.size
    nnz = n * nnz_per_row
    out_data = np.empty(nnz, dtype=data.dtype)
    out_col = np.empty(nnz, dtype=col.dtype)
    out_indptr = np.empty(n + 1, dtype=np.int64)
    out_indptr[0] = 0
    cursor = 0
    for row in range(n):
        for ptr in range(nnz_per_row):
            out_data[cursor] = data[ptr]
            out_col[cursor] = col[ptr]
            cursor += 1
        out_indptr[row + 1] = cursor
    return out_data, out_col, out_indptr


@njit(cache=True)
def _matmul_left_row_major(mat_data, mat_indices, mat_indptr, data, col, indptr):
    nrow_out = mat_indptr.size - 1
    total_nnz = 0
    for out_row in range(nrow_out):
        for aptr in range(mat_indptr[out_row], mat_indptr[out_row + 1]):
            src_row = mat_indices[aptr]
            total_nnz += indptr[src_row + 1] - indptr[src_row]

    out_data = np.empty(total_nnz, dtype=np.float64)
    out_col = np.empty(total_nnz, dtype=np.int64)
    out_indptr = np.empty(nrow_out + 1, dtype=np.int64)
    out_indptr[0] = 0

    cursor = 0
    for out_row in range(nrow_out):
        row_start = cursor
        for aptr in range(mat_indptr[out_row], mat_indptr[out_row + 1]):
            src_row = mat_indices[aptr]
            factor = mat_data[aptr]
            start = indptr[src_row]
            stop = indptr[src_row + 1]
            for jptr in range(start, stop):
                out_data[cursor] = factor * data[jptr]
                out_col[cursor] = col[jptr]
                cursor += 1
        # A row of `mat` (e.g. a divergence operator's cell row, summing
        # several faces) commonly hits more than one `src_row`, and those
        # source rows commonly share a dependency (their own upstream
        # cell's pressure, say) - without this, that dependency is stored
        # once per contributing face instead of once, total.
        cursor = _merge_row_duplicates(out_data, out_col, row_start, cursor)
        out_indptr[out_row + 1] = cursor
    return out_data[:cursor].copy(), out_col[:cursor].copy(), out_indptr


@njit(cache=True)
def _add_row_major(data_a, col_a, indptr_a, data_b, col_b, indptr_b):
    nrow = indptr_a.size - 1
    max_nnz = data_a.size + data_b.size
    out_data = np.empty(max_nnz, dtype=np.float64)
    out_col = np.empty(max_nnz, dtype=np.int64)
    out_indptr = np.empty(nrow + 1, dtype=np.int64)
    out_indptr[0] = 0
    cursor = 0
    for row in range(nrow):
        row_start = cursor
        start = indptr_a[row]
        stop = indptr_a[row + 1]
        for ptr in range(start, stop):
            out_data[cursor] = data_a[ptr]
            out_col[cursor] = col_a[ptr]
            cursor += 1
        start = indptr_b[row]
        stop = indptr_b[row + 1]
        for ptr in range(start, stop):
            out_data[cursor] = data_b[ptr]
            out_col[cursor] = col_b[ptr]
            cursor += 1
        cursor = _merge_row_duplicates(out_data, out_col, row_start, cursor)
        out_indptr[row + 1] = cursor
    return out_data[:cursor].copy(), out_col[:cursor].copy(), out_indptr


@njit(cache=True)
def _group_rows(row, nrows):
    """`(start, order)`: the triplet entries of row `r` are
    `order[start[r]:start[r + 1]]`, in increasing storage order.

    A stable counting sort, once per call. The triplet kernels below used
    to scan every stored entry for every row they wanted - O(rows x nnz),
    which on the 18553-cell Egg model was 0.28 s per divergence and 0.075 s
    per row selection, three quarters of a Newton step. The output is
    unchanged entry for entry: each row still yields its entries in the
    order the scan found them.
    """
    start = np.zeros(nrows + 1, dtype=np.int64)
    for ptr in range(row.size):
        start[row[ptr] + 1] += 1
    for r in range(nrows):
        start[r + 1] += start[r]
    fill = start[:-1].copy()
    order = np.empty(row.size, dtype=np.int64)
    for ptr in range(row.size):
        r = row[ptr]
        order[fill[r]] = ptr
        fill[r] += 1
    return start, order


@njit(cache=True)
def _row_count(row, wanted):
    nrows = 0
    for i in range(row.size):
        nrows = max(nrows, row[i] + 1)
    for i in range(wanted.size):
        nrows = max(nrows, wanted[i] + 1)
    return nrows


@njit(cache=True)
def _take_rows_triplet(data, row, col, rows):
    start, order = _group_rows(row, _row_count(row, rows))
    nnz = 0
    for target in range(rows.size):
        r = rows[target]
        nnz += start[r + 1] - start[r]

    out_data = np.empty(nnz, dtype=data.dtype)
    out_row = np.empty(nnz, dtype=row.dtype)
    out_col = np.empty(nnz, dtype=col.dtype)
    cursor = 0
    for target in range(rows.size):
        r = rows[target]
        for k in range(start[r], start[r + 1]):
            ptr = order[k]
            out_data[cursor] = data[ptr]
            out_row[cursor] = target
            out_col[cursor] = col[ptr]
            cursor += 1
    return out_data, out_row, out_col


@njit(cache=True)
def _row_repeat_triplet(data, col, n):
    nnz_per_row = data.size
    nnz = n * nnz_per_row
    out_data = np.empty(nnz, dtype=data.dtype)
    out_row = np.empty(nnz, dtype=np.int64)
    out_col = np.empty(nnz, dtype=col.dtype)
    cursor = 0
    for r in range(n):
        for ptr in range(nnz_per_row):
            out_data[cursor] = data[ptr]
            out_row[cursor] = r
            out_col[cursor] = col[ptr]
            cursor += 1
    return out_data, out_row, out_col


@njit(cache=True)
def _matmul_left_triplet(mat_data, mat_indices, mat_indptr, data, row, col):
    nrow_out = mat_indptr.size - 1
    start, order = _group_rows(row, _row_count(row, mat_indices))
    total_nnz = 0
    for out_row in range(nrow_out):
        for aptr in range(mat_indptr[out_row], mat_indptr[out_row + 1]):
            src_row = mat_indices[aptr]
            total_nnz += start[src_row + 1] - start[src_row]

    out_data = np.empty(total_nnz, dtype=np.float64)
    out_row = np.empty(total_nnz, dtype=np.int64)
    out_col = np.empty(total_nnz, dtype=np.int64)

    cursor = 0
    for out_r in range(nrow_out):
        row_start = cursor
        for aptr in range(mat_indptr[out_r], mat_indptr[out_r + 1]):
            src_row = mat_indices[aptr]
            factor = mat_data[aptr]
            for k in range(start[src_row], start[src_row + 1]):
                jptr = order[k]
                out_data[cursor] = factor * data[jptr]
                out_row[cursor] = out_r
                out_col[cursor] = col[jptr]
                cursor += 1
        # Same merge as `_matmul_left_row_major` - see that function's
        # comment. `out_row` is uniformly `out_r` across this slice
        # already, so only `data`/`col` need compacting.
        cursor = _merge_row_duplicates(out_data, out_col, row_start, cursor)
    return out_data[:cursor].copy(), out_row[:cursor].copy(), out_col[:cursor].copy()


@njit(cache=True)
def _vstack_triplet(data_parts, row_parts, col_parts, nrows):
    total_nnz = 0
    total_rows = 0
    for i in range(len(data_parts)):
        total_nnz += data_parts[i].size
        total_rows += nrows[i]
    out_data = np.empty(total_nnz, dtype=np.float64)
    out_row = np.empty(total_nnz, dtype=np.int64)
    out_col = np.empty(total_nnz, dtype=np.int64)
    cursor = 0
    row_offset = 0
    for i in range(len(data_parts)):
        data = data_parts[i]
        row = row_parts[i]
        col = col_parts[i]
        for ptr in range(data.size):
            out_data[cursor] = data[ptr]
            out_row[cursor] = row[ptr] + row_offset
            out_col[cursor] = col[ptr]
            cursor += 1
        row_offset += nrows[i]
    return out_data, out_row, out_col, total_rows


@njit(cache=True)
def _vstack_row_major(data_parts, col_parts, indptr_parts, nrows):
    total_nnz = 0
    total_rows = 0
    for i in range(len(data_parts)):
        total_nnz += data_parts[i].size
        total_rows += nrows[i]

    out_data = np.empty(total_nnz, dtype=np.float64)
    out_col = np.empty(total_nnz, dtype=np.int64)
    out_indptr = np.empty(total_rows + 1, dtype=np.int64)
    out_indptr[0] = 0
    cursor = 0
    out_row = 0
    for i in range(len(data_parts)):
        data = data_parts[i]
        col = col_parts[i]
        indptr = indptr_parts[i]
        for r in range(nrows[i]):
            start = indptr[r]
            stop = indptr[r + 1]
            for ptr in range(start, stop):
                out_data[cursor] = data[ptr]
                out_col[cursor] = col[ptr]
                cursor += 1
            out_row += 1
            out_indptr[out_row] = cursor
    return out_data, out_col, out_indptr, total_rows


def left_scale_row_major(data, indptr, scale):
    """Scale each row of row-major diagonal data."""
    return _left_scale_row_major(
        np.asarray(data, dtype=np.float64),
        np.asarray(indptr, dtype=np.int64),
        np.asarray(scale, dtype=np.float64),
    )


def take_rows_row_major(data, col, indptr, rows):
    """Select rows from row-major diagonal data."""
    return _take_rows_row_major(
        np.asarray(data, dtype=np.float64),
        np.asarray(col, dtype=np.int64),
        np.asarray(indptr, dtype=np.int64),
        np.asarray(rows, dtype=np.int64),
    )


def row_repeat_row_major(data, col, n):
    """Repeat a one-row row-major diagonal block."""
    return _row_repeat_row_major(
        np.asarray(data, dtype=np.float64),
        np.asarray(col, dtype=np.int64),
        int(n),
    )


def matmul_left_row_major(mat_data, mat_indices, mat_indptr, data, col, indptr):
    """Left-multiply row-major diagonal data by a CSR matrix."""
    return _matmul_left_row_major(
        np.asarray(mat_data, dtype=np.float64),
        np.asarray(mat_indices, dtype=np.int64),
        np.asarray(mat_indptr, dtype=np.int64),
        np.asarray(data, dtype=np.float64),
        np.asarray(col, dtype=np.int64),
        np.asarray(indptr, dtype=np.int64),
    )


def add_row_major(data_a, col_a, indptr_a, data_b, col_b, indptr_b):
    """Add two row-major diagonal blocks without sparse assembly."""
    return _add_row_major(
        np.asarray(data_a, dtype=np.float64),
        np.asarray(col_a, dtype=np.int64),
        np.asarray(indptr_a, dtype=np.int64),
        np.asarray(data_b, dtype=np.float64),
        np.asarray(col_b, dtype=np.int64),
        np.asarray(indptr_b, dtype=np.int64),
    )


def take_rows_triplet(data, row, col, rows):
    """Select rows from triplet diagonal data."""
    return _take_rows_triplet(
        np.asarray(data, dtype=np.float64),
        np.asarray(row, dtype=np.int64),
        np.asarray(col, dtype=np.int64),
        np.asarray(rows, dtype=np.int64),
    )


def row_repeat_triplet(data, col, n):
    """Repeat a one-row triplet diagonal block."""
    return _row_repeat_triplet(
        np.asarray(data, dtype=np.float64),
        np.asarray(col, dtype=np.int64),
        int(n),
    )


def matmul_left_triplet(mat_data, mat_indices, mat_indptr, data, row, col):
    """Left-multiply triplet diagonal data by a CSR matrix."""
    return _matmul_left_triplet(
        np.asarray(mat_data, dtype=np.float64),
        np.asarray(mat_indices, dtype=np.int64),
        np.asarray(mat_indptr, dtype=np.int64),
        np.asarray(data, dtype=np.float64),
        np.asarray(row, dtype=np.int64),
        np.asarray(col, dtype=np.int64),
    )


def vstack_triplet(data_parts, row_parts, col_parts, nrows):
    """Vertically stack triplet diagonal blocks."""
    return _vstack_triplet(
        tuple(np.asarray(x, dtype=np.float64) for x in data_parts),
        tuple(np.asarray(x, dtype=np.int64) for x in row_parts),
        tuple(np.asarray(x, dtype=np.int64) for x in col_parts),
        np.asarray(nrows, dtype=np.int64),
    )


def vstack_row_major(data_parts, col_parts, indptr_parts, nrows):
    """Vertically stack row-major diagonal blocks."""
    return _vstack_row_major(
        tuple(np.asarray(x, dtype=np.float64) for x in data_parts),
        tuple(np.asarray(x, dtype=np.int64) for x in col_parts),
        tuple(np.asarray(x, dtype=np.int64) for x in indptr_parts),
        np.asarray(nrows, dtype=np.int64),
    )


__all__ = [
    "HAS_NUMBA_KERNELS",
    "add_row_major",
    "left_scale_row_major",
    "matmul_left_row_major",
    "matmul_left_triplet",
    "row_repeat_row_major",
    "row_repeat_triplet",
    "take_rows_row_major",
    "take_rows_triplet",
    "vstack_row_major",
    "vstack_triplet",
]


if __name__ == "__main__":
    import scipy.sparse as sp

    def _reference_take(data, row, col, rows):
        picked = [(data[p], t, col[p]) for t, r in enumerate(rows)
                  for p in range(row.size) if row[p] == r]
        return [np.array([x[i] for x in picked]) for i in range(3)]

    def test_take_rows_keeps_storage_order_and_repeats():
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        row = np.array([2, 0, 2, 1, 0])
        col = np.array([0, 1, 2, 3, 4])
        rows = np.array([0, 2, 0, 3])
        out = take_rows_triplet(data, row, col, rows)
        for got, want in zip(out, _reference_take(data, row, col, rows)):
            np.testing.assert_array_equal(got, want)
        print("ok test_take_rows_keeps_storage_order_and_repeats")

    def test_matmul_left_equals_the_dense_product():
        rng = np.random.default_rng(0)
        n = 50
        row, col = rng.integers(0, n, 3*n), rng.integers(0, n, 3*n)
        data = rng.standard_normal(3*n)
        M = sp.random(n + 5, n, density=0.1, random_state=1, format="csr")
        d, r, c = matmul_left_triplet(M.data, M.indices, M.indptr, data, row, col)
        got = sp.coo_matrix((d, (r, c)), shape=(n + 5, n)).toarray()
        J = sp.coo_matrix((data, (row, col)), shape=(n, n)).toarray()
        np.testing.assert_allclose(got, M.toarray() @ J, atol=1e-13)
        print("ok test_matmul_left_equals_the_dense_product")

    test_take_rows_keeps_storage_order_and_repeats()
    test_matmul_left_equals_the_dense_product()
    print("\nAll tests passed!")
