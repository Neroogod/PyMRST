"""Repair zero diagonal entries by reordering equations for ILU."""

from __future__ import annotations

import warnings

import numpy as np
import scipy.sparse as sp


def reorder_for_ilu(matrix, rhs, number_of_cells=0):
    """Swap rows after ``number_of_cells`` to obtain nonzero diagonal entries."""
    sparse = sp.issparse(matrix)
    array = matrix.toarray() if sparse else np.asarray(matrix)
    rhs_array = np.asarray(rhs)
    offset = int(number_of_cells)
    sub = array[offset:, offset:]
    bad = np.flatnonzero(np.diag(sub) == 0)
    if bad.size == 0:
        return matrix.copy() if hasattr(matrix, "copy") else array.copy(), rhs_array.copy()
    rows, columns = np.nonzero(sub)
    replacements = np.full(bad.size, -1, dtype=np.int64)
    for index, entry in enumerate(bad):
        possible = columns[rows == entry]
        candidates = rows[columns == entry]
        common = np.intersect1d(possible, candidates)
        if common.size == 0:
            warnings.warn(
                "Unable to reorder equations, zeros on the diagonal still present",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        replacements[index] = common[0]
        columns[rows == common[0]] = -1
    order = np.arange(array.shape[0])
    valid = replacements >= 0
    order[offset + replacements[valid]] = offset + bad[valid]
    order[offset + bad[valid]] = offset + replacements[valid]
    reordered = array[order, :]
    if sparse:
        reordered = type(matrix)(reordered)
    return reordered, rhs_array[order, ...]


reorderForILU = reorder_for_ilu

__all__ = ["reorder_for_ilu", "reorderForILU"]
