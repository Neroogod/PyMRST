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
        for aptr in range(mat_indptr[out_row], mat_indptr[out_row + 1]):
            src_row = mat_indices[aptr]
            factor = mat_data[aptr]
            start = indptr[src_row]
            stop = indptr[src_row + 1]
            for jptr in range(start, stop):
                out_data[cursor] = factor * data[jptr]
                out_col[cursor] = col[jptr]
                cursor += 1
        out_indptr[out_row + 1] = cursor
    return out_data, out_col, out_indptr


@njit(cache=True)
def _add_row_major(data_a, col_a, indptr_a, data_b, col_b, indptr_b):
    nrow = indptr_a.size - 1
    nnz = data_a.size + data_b.size
    out_data = np.empty(nnz, dtype=np.float64)
    out_col = np.empty(nnz, dtype=np.int64)
    out_indptr = np.empty(nrow + 1, dtype=np.int64)
    out_indptr[0] = 0
    cursor = 0
    for row in range(nrow):
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
        out_indptr[row + 1] = cursor
    return out_data, out_col, out_indptr


@njit(cache=True)
def _take_rows_triplet(data, row, col, rows):
    keep = np.zeros(row.size, dtype=np.bool_)
    nnz = 0
    for target in range(rows.size):
        r = rows[target]
        for ptr in range(row.size):
            if row[ptr] == r:
                keep[ptr] = True
                nnz += 1

    out_data = np.empty(nnz, dtype=data.dtype)
    out_row = np.empty(nnz, dtype=row.dtype)
    out_col = np.empty(nnz, dtype=col.dtype)
    cursor = 0
    for target in range(rows.size):
        r = rows[target]
        for ptr in range(row.size):
            if keep[ptr] and row[ptr] == r:
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
    total_nnz = 0
    for out_row in range(nrow_out):
        for aptr in range(mat_indptr[out_row], mat_indptr[out_row + 1]):
            src_row = mat_indices[aptr]
            for jptr in range(row.size):
                if row[jptr] == src_row:
                    total_nnz += 1

    out_data = np.empty(total_nnz, dtype=np.float64)
    out_row = np.empty(total_nnz, dtype=np.int64)
    out_col = np.empty(total_nnz, dtype=np.int64)

    cursor = 0
    for out_r in range(nrow_out):
        for aptr in range(mat_indptr[out_r], mat_indptr[out_r + 1]):
            src_row = mat_indices[aptr]
            factor = mat_data[aptr]
            for jptr in range(row.size):
                if row[jptr] == src_row:
                    out_data[cursor] = factor * data[jptr]
                    out_row[cursor] = out_r
                    out_col[cursor] = col[jptr]
                    cursor += 1
    return out_data, out_row, out_col


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
