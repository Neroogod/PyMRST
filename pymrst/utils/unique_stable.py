"""Stable unique values or rows (MRST ``utils/uniqueStable.m``)."""

from __future__ import annotations

import numpy as np


def unique_stable(a, rows: bool = False):
    """Return ``(c, ia, ic)`` in first-appearance order.

    Indices are zero-based.  With ``rows=False`` input is linearized in
    Fortran order, matching MATLAB's column-major element traversal.
    """
    arr = np.asarray(a)
    if rows and arr.ndim == 1:
        if arr.size == 0:
            empty = np.empty(0, dtype=np.int64)
            return arr.copy(), empty, empty
        return arr.copy(), np.array([0], dtype=np.int64), \
            np.array([0], dtype=np.int64)
    source = arr if rows else arr.reshape(-1, order="F")
    if source.size == 0:
        empty = np.empty(0, dtype=np.int64)
        return source.copy(), empty, empty

    keys = [tuple(row.tolist()) for row in source] if rows else source.tolist()
    lookup, values, first, inverse = {}, [], [], []
    for position, key in enumerate(keys):
        # Treat all NaNs as the same value, as MATLAB unique does.
        if rows:
            key = tuple(("__nan__" if isinstance(v, float) and np.isnan(v)
                         else v) for v in key)
        elif isinstance(key, float) and np.isnan(key):
            key = "__nan__"
        if key not in lookup:
            lookup[key] = len(values)
            values.append(source[position].copy() if rows else source[position])
            first.append(position)
        inverse.append(lookup[key])
    c = np.asarray(values, dtype=arr.dtype)
    return c, np.asarray(first, dtype=np.int64), np.asarray(inverse, dtype=np.int64)


uniqueStable = unique_stable

__all__ = ["unique_stable", "uniqueStable"]
