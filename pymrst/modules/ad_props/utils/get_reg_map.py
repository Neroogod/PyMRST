"""Map values onto property regions (MRST ``ad-props/utils/getRegMap.m``).

Returns, for a value array and a region partition, the indices of the entries
belonging to each region. Every regional property function starts here: it
splits its input by region, evaluates one table per region, and scatters the
results back.

Region numbers are **0-based** throughout PyMRST, converted once at the deck
boundary. MRST's are 1-based, so any MATLAB reference dump must convert on the
MATLAB side.
"""

from typing import Any, List, Optional, Sequence, Union

import numpy as np

# The whole domain, when there is only one region. `slice(None)` indexes a
# NumPy array exactly as MRST's ':' indexes a MATLAB one.
ALL = slice(None)

RegionIndex = Union[slice, np.ndarray]


def _value_count(val) -> int:
    """Number of entries in `val`, without assuming it is a plain array."""
    if val is None:
        return 0
    inner = getattr(val, "val", val)          # ADI carries its value in `.val`
    return int(np.size(inner))


def get_reg_map(
    val,
    reg_num: Optional[np.ndarray],
    reg_inx: Optional[Sequence[Any]],
    *,
    cell_inx: Optional[np.ndarray] = None,
) -> List[RegionIndex]:
    """Return one index array per region for the entries of ``val``.

    ``reg_num`` is the per-cell region of the full domain and ``reg_inx`` the
    precomputed per-region cell lists. ``cell_inx`` restricts the mapping to a
    subset of cells, which is how a property gets evaluated on part of the
    domain - a well's perforations, say - rather than all of it.
    """
    reg_num = None if reg_num is None else np.asarray(reg_num).ravel()
    # ALL is the unpartitioned case - MRST's ':' - and means one region, so it
    # has no length to take. Callers reach here whenever a property is
    # evaluated on a deck with no REGIONS section.
    n_regions = (1 if reg_inx is None or reg_inx is ALL
                 or isinstance(reg_inx, slice) else len(reg_inx))

    # A scalar region number means every value belongs to that one region.
    if reg_num is not None and reg_num.size == 1:
        out: List[RegionIndex] = [np.zeros(0, dtype=np.int64)
                                  for _ in range(max(n_regions,
                                                     int(reg_num[0]) + 1))]
        out[int(reg_num[0])] = np.arange(_value_count(val), dtype=np.int64)
        return out

    count = (_value_count(val) if val is not None and _value_count(val)
             else (reg_num.size if cell_inx is None and reg_num is not None
                   else np.size(cell_inx)))

    if cell_inx is None:
        if n_regions == 1:
            return [ALL]
        if reg_num is not None and count != reg_num.size:
            raise ValueError("region reference for input undefined: "
                             f"{count} values against {reg_num.size} cells")
        return [np.asarray(inx) for inx in reg_inx]

    cell_inx = np.atleast_1d(np.asarray(cell_inx, dtype=np.int64)).ravel()
    if reg_num is None or reg_num.size == 0:
        # No partition means one region covering every requested cell.
        return [np.arange(cell_inx.size, dtype=np.int64)]

    local = reg_num[cell_inx]
    if cell_inx.size > 1:
        if count != cell_inx.size:
            raise ValueError("number of cell indices must match the number of "
                             f"values: {cell_inx.size} against {count}")
        return [np.flatnonzero(local == region) for region in range(n_regions)]

    # A single cell is allowed, so a property can be explored one cell at a time.
    out = [np.zeros(0, dtype=np.int64) for _ in range(n_regions)]
    out[int(local[0])] = np.arange(count, dtype=np.int64)
    return out


getRegMap = get_reg_map


if __name__ == "__main__":

    def test_single_region_covers_everything() -> None:
        assert get_reg_map(np.zeros(5), None, [ALL]) == [ALL]
        print("ok test_single_region_covers_everything")

    def test_two_regions_split_the_domain() -> None:
        reg_num = np.array([0, 0, 1, 1, 0])
        reg_inx = [np.flatnonzero(reg_num == r) for r in range(2)]
        out = get_reg_map(np.zeros(5), reg_num, reg_inx)
        assert np.array_equal(out[0], [0, 1, 4])
        assert np.array_equal(out[1], [2, 3])
        print("ok test_two_regions_split_the_domain")

    def test_every_entry_lands_in_exactly_one_region() -> None:
        rng = np.random.default_rng(0)
        reg_num = rng.integers(0, 3, 20)
        reg_inx = [np.flatnonzero(reg_num == r) for r in range(3)]
        out = get_reg_map(np.zeros(20), reg_num, reg_inx)
        stacked = np.concatenate(out)
        assert np.array_equal(np.sort(stacked), np.arange(20))
        print("ok test_every_entry_lands_in_exactly_one_region")

    def test_cell_subset_indexes_the_subset_not_the_domain() -> None:
        """The indices returned address `val`, not the global cell numbering."""
        reg_num = np.array([0, 1, 0, 1, 1])
        reg_inx = [np.flatnonzero(reg_num == r) for r in range(2)]
        cells = np.array([1, 2, 4])              # regions 1, 0, 1
        out = get_reg_map(np.zeros(3), reg_num, reg_inx, cell_inx=cells)
        assert np.array_equal(out[0], [1]), out
        assert np.array_equal(out[1], [0, 2]), out
        print("ok test_cell_subset_indexes_the_subset_not_the_domain")

    def test_a_single_cell_is_allowed() -> None:
        reg_num = np.array([0, 1, 1])
        out = get_reg_map(np.zeros(1), reg_num, [None, None],
                          cell_inx=np.array([2]))
        assert np.array_equal(out[1], [0])
        assert out[0].size == 0
        print("ok test_a_single_cell_is_allowed")

    def test_absent_partition_gives_one_region() -> None:
        out = get_reg_map(np.zeros(4), None, [None],
                          cell_inx=np.array([3, 4, 5, 6]))
        assert np.array_equal(out[0], [0, 1, 2, 3])
        print("ok test_absent_partition_gives_one_region")

    def test_mismatched_lengths_are_rejected() -> None:
        reg_num = np.array([0, 1, 0, 1])
        reg_inx = [np.flatnonzero(reg_num == r) for r in range(2)]
        try:
            get_reg_map(np.zeros(3), reg_num, reg_inx)
        except ValueError:
            print("ok test_mismatched_lengths_are_rejected")
            return
        raise AssertionError("a value/cell count mismatch must be rejected")

    test_single_region_covers_everything()
    test_two_regions_split_the_domain()
    test_every_entry_lands_in_exactly_one_region()
    test_cell_subset_indexes_the_subset_not_the_domain()
    test_a_single_cell_is_allowed()
    def test_all_is_accepted_as_the_unpartitioned_index() -> None:
        """`Regions()` leaves SATINX as ALL, which has no length to take.

        Every property evaluated on a deck with no REGIONS section arrives
        here that way, so treating ALL as "one region" is the common path, not
        an edge case.
        """
        out = get_reg_map(np.zeros(5), None, ALL)
        assert out == [ALL], out
        print("ok test_all_is_accepted_as_the_unpartitioned_index")

    test_absent_partition_gives_one_region()
    test_all_is_accepted_as_the_unpartitioned_index()
    test_mismatched_lengths_are_rejected()
    print("\nAll tests passed!")
