"""Flat table extension (MRST ``ad-props/utils/extendTab.m``).

Adds one row at each end of a table, repeating the end values at an abscissa
``pm`` beyond the table's own range. The extra rows are flat, so a lookup
outside the table returns the nearest tabulated value instead of extrapolating
a trend that the deck never stated.

That matters most for saturation functions: a Newton step routinely proposes
saturations slightly outside [0, 1], and a linearly extrapolated relative
permeability goes negative there, which is worse than useless. Flat extension
keeps the property in range so the step can be rejected on its merits.

A list of tables is extended element-wise, matching MRST's cell-array branch -
saturation and PVT keywords hold one table per region and are passed whole.
"""

from typing import List, Union

import numpy as np

TableOrTables = Union[np.ndarray, List[np.ndarray]]


def _extend(tab: np.ndarray, pm: float) -> np.ndarray:
    tab = np.asarray(tab, dtype=float)
    first = tab[0].copy()
    first[0] -= pm
    last = tab[-1].copy()
    last[0] += pm
    return np.vstack([first, tab, last])


def extend_tab(tab: TableOrTables, pm: float = 1.0) -> TableOrTables:
    """Pad ``tab`` with flat rows at ``x[0] - pm`` and ``x[-1] + pm``."""
    if isinstance(tab, (list, tuple)):
        return [_extend(t, pm) for t in tab]
    return _extend(tab, pm)


extendTab = extend_tab

__all__ = ["extend_tab", "extendTab"]


if __name__ == "__main__":

    TABLE = np.array([[0.2, 0.0, 1.0],
                      [0.5, 0.3, 0.4],
                      [0.8, 1.0, 0.0]])

    def test_two_rows_are_added() -> None:
        out = extend_tab(TABLE)
        assert out.shape == (5, 3), out.shape
        print("ok test_two_rows_are_added")

    def test_the_original_rows_are_unchanged_in_the_middle() -> None:
        out = extend_tab(TABLE)
        assert np.allclose(out[1:-1], TABLE)
        print("ok test_the_original_rows_are_unchanged_in_the_middle")

    def test_only_the_abscissa_moves() -> None:
        out = extend_tab(TABLE)
        assert np.isclose(out[0, 0], 0.2 - 1.0)
        assert np.allclose(out[0, 1:], TABLE[0, 1:]), "values repeat, flat"
        assert np.isclose(out[-1, 0], 0.8 + 1.0)
        assert np.allclose(out[-1, 1:], TABLE[-1, 1:])
        print("ok test_only_the_abscissa_moves")

    def test_pm_sets_how_far_the_extension_reaches() -> None:
        out = extend_tab(TABLE, pm=0.05)
        assert np.isclose(out[0, 0], 0.15) and np.isclose(out[-1, 0], 0.85)
        print("ok test_pm_sets_how_far_the_extension_reaches")

    def test_lookups_outside_the_table_stay_in_range() -> None:
        """The point of the extension: no negative relative permeability."""
        out = extend_tab(TABLE, pm=0.5)
        below = np.interp(-0.1, out[:, 0], out[:, 1])
        above = np.interp(1.1, out[:, 0], out[:, 1])
        assert np.isclose(below, TABLE[0, 1]), below
        assert np.isclose(above, TABLE[-1, 1]), above
        extrapolated = np.interp(-0.1, TABLE[:, 0], TABLE[:, 1])
        assert extrapolated >= 0.0, "np.interp clamps; the deck table does not"
        print("ok test_lookups_outside_the_table_stay_in_range")

    def test_a_list_of_tables_is_extended_element_wise() -> None:
        out = extend_tab([TABLE, TABLE * 2.0])
        assert isinstance(out, list) and len(out) == 2
        assert out[0].shape == (5, 3) and out[1].shape == (5, 3)
        assert np.isclose(out[1][0, 0], 0.4 - 1.0)
        print("ok test_a_list_of_tables_is_extended_element_wise")

    def test_the_input_is_not_modified() -> None:
        original = TABLE.copy()
        extend_tab(TABLE)
        assert np.allclose(TABLE, original)
        print("ok test_the_input_is_not_modified")

    def test_a_single_row_table_is_extended_both_ways() -> None:
        out = extend_tab(np.array([[0.5, 0.7]]))
        assert out.shape == (3, 2)
        assert np.allclose(out[:, 1], 0.7), "flat everywhere"
        print("ok test_a_single_row_table_is_extended_both_ways")

    test_two_rows_are_added()
    test_the_original_rows_are_unchanged_in_the_middle()
    test_only_the_abscissa_moves()
    test_pm_sets_how_far_the_extension_reaches()
    test_lookups_outside_the_table_stay_in_range()
    test_a_list_of_tables_is_extended_element_wise()
    test_the_input_is_not_modified()
    test_a_single_row_table_is_extended_both_ways()
    print("\nAll tests passed!")
