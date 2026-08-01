"""Expand matrix columns to a cell/list layout (MRST ``expandMatrixToCell.m``)."""

from __future__ import annotations

from typing import Any

import numpy as np


def expandMatrixToCell(matrix: Any, subset: Any = slice(None)) -> list[Any]:
    """Return one column vector per matrix column.

    This mirrors MRST's small ``expandMatrixToCell`` utility. If ``matrix`` is
    already a Python list/tuple (the closest analogue of a MATLAB cell array), it
    is returned as a list unchanged. Otherwise, a 2-D array is split by columns;
    a 1-D array is treated as one column.
    """
    if isinstance(matrix, (list, tuple)):
        return list(matrix)
    arr = np.asarray(matrix)
    if subset == ":":
        subset = slice(None)
    if arr.ndim == 0:
        arr = arr.reshape(1, 1)
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError("matrix must be one- or two-dimensional")
    return [arr[subset, i] for i in range(arr.shape[1])]


def expand_matrix_to_cell(matrix: Any, subset: Any = slice(None)) -> list[Any]:
    """Pythonic alias for :func:`expandMatrixToCell`."""
    return expandMatrixToCell(matrix, subset=subset)


__all__ = ["expandMatrixToCell", "expand_matrix_to_cell"]
