"""Packed-data editing helpers used by gridprocessing private ports."""

from typing import Tuple

import numpy as np

from pymrst.utils.gridtools import mcolon


def insert_in_packed_data(
    pos: np.ndarray,
    data: np.ndarray,
    rows: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Append `values[i]` to packed segment `rows[i]`."""
    n_segments = pos.size - 1
    order = np.lexsort((values, rows))
    sorted_values = values[order]
    added = np.bincount(rows, minlength=n_segments).astype(np.int64)
    counts = np.diff(pos)
    new_pos = np.concatenate([[0], np.cumsum(counts + added)]).astype(np.int64)
    unique_rows = np.unique(rows)
    inserted_ix = mcolon(new_pos[unique_rows] + counts[unique_rows], new_pos[unique_rows + 1])
    new_data = np.zeros(data.size + values.size, dtype=np.int64)
    keep = np.ones(new_data.size, dtype=bool)
    keep[inserted_ix] = False
    new_data[keep] = data
    new_data[inserted_ix] = sorted_values
    return new_data, new_pos


def remove_from_packed_data(
    pos: np.ndarray,
    data: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop every packed-data entry whose value is in `values`."""
    n_segments = pos.size - 1
    if data.size == 0 or np.asarray(values).size == 0:
        return data.copy(), pos.copy()
    mark = np.zeros(int(data.max()) + 1, dtype=bool)
    mark[values] = True
    hit = mark[data]
    segment = np.repeat(np.arange(n_segments, dtype=np.int64), np.diff(pos))
    lost = np.bincount(segment[hit], minlength=n_segments).astype(np.int64)
    return data[~hit], pos - np.concatenate([[0], np.cumsum(lost)])


def map_excluding(removed: np.ndarray) -> np.ndarray:
    """Return old-index to new-index map after removing marked entries."""
    mapping = np.cumsum(~removed).astype(np.int64) - 1
    mapping[removed] = -1
    return mapping

