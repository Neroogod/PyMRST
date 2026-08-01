"""Packed-data editing helpers used by gridprocessing private ports."""

from typing import Tuple

import numpy as np

from pymrst.utils.mcolon import mcolon


def insert_in_packed_data(
    pos: np.ndarray,
    data: np.ndarray,
    rows: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Append `values[i]` to packed segment `rows[i]`."""
    new_data, new_pos, _ = insert_in_packed_data_layout(pos, data, rows, values)
    return new_data, new_pos


def insert_in_packed_data_layout(
    pos: np.ndarray,
    data: np.ndarray,
    rows: np.ndarray,
    values: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """`insert_in_packed_data`, also reporting where each entry ended up.

    The layout is ``(keep, inserted, order)``: existing entries land on
    ``keep``, and ``values[order]`` lands on ``inserted``. Arrays running
    parallel to the packed data - cell-face tags, say - are moved with the
    same layout so they stay aligned.
    """
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
    return new_data, new_pos, (keep, inserted_ix, order)


def place_with_layout(
    values: np.ndarray,
    new_values: np.ndarray,
    layout: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> np.ndarray:
    """Rebuild a packed-parallel array under an `insert_in_packed_data` layout."""
    keep, inserted_ix, order = layout
    values = np.asarray(values)
    new_values = np.asarray(new_values, dtype=values.dtype)
    shape = (keep.size,) + values.shape[1:]
    combined = np.zeros(shape, dtype=values.dtype)
    combined[keep] = values
    combined[inserted_ix] = new_values[order]
    return combined


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


if __name__ == "__main__":

    def test_insert_appends_to_the_named_segments() -> None:
        pos = np.array([0, 2, 4], dtype=np.int64)
        data = np.array([0, 1, 2, 3], dtype=np.int64)
        new_data, new_pos = insert_in_packed_data(
            pos, data, np.array([0, 1]), np.array([9, 8])
        )
        assert np.array_equal(new_pos, [0, 3, 6]), new_pos
        assert np.array_equal(new_data, [0, 1, 9, 2, 3, 8]), new_data
        print("ok test_insert_appends_to_the_named_segments")

    def test_layout_places_parallel_arrays_alongside() -> None:
        pos = np.array([0, 2, 4], dtype=np.int64)
        data = np.array([0, 1, 2, 3], dtype=np.int64)
        new_data, _, layout = insert_in_packed_data_layout(
            pos, data, np.array([0, 1]), np.array([9, 8])
        )
        parallel = place_with_layout(
            np.array([10, 11, 12, 13]), np.array([99, 88]), layout
        )
        # the parallel entry must sit wherever its packed value landed
        assert parallel[new_data == 9].item() == 99, parallel
        assert parallel[new_data == 8].item() == 88, parallel
        assert np.array_equal(parallel[np.isin(new_data, [0, 1, 2, 3])],
                              [10, 11, 12, 13]), parallel
        print("ok test_layout_places_parallel_arrays_alongside")

    def test_layout_carries_multi_column_values() -> None:
        pos = np.array([0, 1], dtype=np.int64)
        _, _, layout = insert_in_packed_data_layout(
            pos, np.array([5], dtype=np.int64), np.array([0]), np.array([7])
        )
        parallel = place_with_layout(
            np.array([[1.0, 2.0]]), np.array([[3.0, 4.0]]), layout
        )
        assert parallel.shape == (2, 2), parallel.shape
        assert np.array_equal(np.sort(parallel, axis=0), [[1.0, 2.0], [3.0, 4.0]])
        print("ok test_layout_carries_multi_column_values")

    def test_remove_drops_entries_and_shifts_positions() -> None:
        pos = np.array([0, 2, 4], dtype=np.int64)
        data = np.array([0, 1, 2, 3], dtype=np.int64)
        new_data, new_pos = remove_from_packed_data(pos, data, np.array([1]))
        assert np.array_equal(new_data, [0, 2, 3]), new_data
        assert np.array_equal(new_pos, [0, 1, 3]), new_pos
        print("ok test_remove_drops_entries_and_shifts_positions")

    def test_map_excluding_marks_removed_entries() -> None:
        mapping = map_excluding(np.array([False, True, False]))
        assert np.array_equal(mapping, [0, -1, 1]), mapping
        print("ok test_map_excluding_marks_removed_entries")

    test_insert_appends_to_the_named_segments()
    test_layout_places_parallel_arrays_alongside()
    test_layout_carries_multi_column_values()
    test_remove_drops_entries_and_shifts_positions()
    test_map_excluding_marks_removed_entries()
    print("\nAll tests passed!")
