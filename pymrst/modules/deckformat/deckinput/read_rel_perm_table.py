"""Saturation-function tables (MRST ``deckinput/readRelPermTable.m``).

Reads the `ntab` tables behind SWOF, SGOF, SWFN and friends. Two deck
conventions are handled:

- a `1*` entry defaults a single value, which is filled by linear
  interpolation against the table's first column;
- an entirely empty table repeats the previous one. The *first* table cannot
  be defaulted, because there is nothing to repeat - MRST raises there and so
  does this.
"""

from typing import List, Optional

import numpy as np

from pymrst.modules.deckformat.deckinput.read_defaulted_record import (
    DeckCursor,
    strip_quotes,
)


def _to_float(token: str) -> float:
    """A deck number, with `1*` meaning "defaulted" rather than a value."""
    token = strip_quotes(token)
    if token.endswith("*") and token[:-1].isdigit():
        return np.nan
    return float(token)


def _fill_defaults(table: np.ndarray) -> np.ndarray:
    """Replace NaN entries by linear interpolation against column 0."""
    x = table[:, 0]
    for column in range(1, table.shape[1]):
        values = table[:, column]
        missing = np.isnan(values)
        if not np.any(missing) or np.all(missing):
            continue
        known = ~missing
        values[missing] = np.interp(x[missing], x[known], values[known])
    return table


def read_rel_perm_table(cursor: DeckCursor, keyword: str, n_tab: int,
                        n_col: int) -> List[np.ndarray]:
    """Read ``n_tab`` saturation-function tables of ``n_col`` columns."""
    tables: List[np.ndarray] = []
    for index in range(n_tab):
        fields = cursor.read_record_tokens()
        if not fields:
            if index == 0:
                raise ValueError(
                    f"the first table in saturation function keyword "
                    f"'{keyword}' cannot be defaulted"
                )
            tables.append(tables[-1].copy())
            continue
        values = np.array([_to_float(f) for f in fields], dtype=float)
        if values.size % n_col:
            raise ValueError(
                f"{keyword}: {values.size} values do not fill rows of {n_col}"
            )
        tables.append(_fill_defaults(values.reshape(-1, n_col)))
    return tables


readRelPermTable = read_rel_perm_table


if __name__ == "__main__":
    from pymrst.modules.deckformat.deckinput.read_defaulted_record import tokenize

    def test_reads_one_table_of_four_columns() -> None:
        cursor = DeckCursor(tokenize("0.1 0.0 1.0 0.0\n0.9 1.0 0.0 0.0 /\n"))
        tables = read_rel_perm_table(cursor, "SWOF", 1, 4)
        assert len(tables) == 1 and tables[0].shape == (2, 4)
        assert np.allclose(tables[0][:, 0], [0.1, 0.9])
        print("ok test_reads_one_table_of_four_columns")

    def test_reads_several_tables_in_sequence() -> None:
        text = "0.1 0.0 /\n0.2 1.0 /\n"
        tables = read_rel_perm_table(DeckCursor(tokenize(text)), "SWFN", 2, 2)
        assert len(tables) == 2
        assert np.allclose(tables[0][0, 0], 0.1)
        assert np.allclose(tables[1][0, 0], 0.2)
        print("ok test_reads_several_tables_in_sequence")

    def test_defaulted_entry_is_interpolated() -> None:
        """`1*` in the middle column is filled from its neighbours."""
        text = "0.0 0.0\n1.0 1* \n2.0 20.0 /\n"
        table = read_rel_perm_table(DeckCursor(tokenize(text)), "SWFN", 1, 2)[0]
        assert np.isclose(table[1, 1], 10.0), table
        print("ok test_defaulted_entry_is_interpolated")

    def test_an_empty_table_repeats_the_previous_one() -> None:
        text = "0.1 0.5 /\n/\n"
        tables = read_rel_perm_table(DeckCursor(tokenize(text)), "SWOF", 2, 2)
        assert np.allclose(tables[0], tables[1])
        assert tables[0] is not tables[1], "must be a copy, not an alias"
        print("ok test_an_empty_table_repeats_the_previous_one")

    def test_the_first_table_cannot_be_defaulted() -> None:
        try:
            read_rel_perm_table(DeckCursor(tokenize("/\n")), "SWOF", 1, 2)
        except ValueError as error:
            assert "cannot be defaulted" in str(error)
            print("ok test_the_first_table_cannot_be_defaulted")
            return
        raise AssertionError("a defaulted first table must be rejected")

    def test_a_ragged_table_is_rejected() -> None:
        try:
            read_rel_perm_table(DeckCursor(tokenize("1 2 3 /\n")), "SWOF", 1, 2)
        except ValueError as error:
            assert "rows of 2" in str(error)
            print("ok test_a_ragged_table_is_rejected")
            return
        raise AssertionError("a ragged table must be rejected")

    test_reads_one_table_of_four_columns()
    test_reads_several_tables_in_sequence()
    test_defaulted_entry_is_interpolated()
    test_an_empty_table_repeats_the_previous_one()
    test_the_first_table_cannot_be_defaulted()
    test_a_ragged_table_is_rejected()
    print("\nAll tests passed!")
