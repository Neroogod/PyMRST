"""Read a deck's SOLUTION section (MRST ``deckinput/readSOLUTION.m``).

The section says what the reservoir contains before the first time step, in
one of two mutually exclusive ways:

* **`EQUIL`** - a datum depth and pressure plus the fluid contacts, from which
  the initial state is *computed* by integrating the hydrostatic gradient. This
  is what nearly every deck writes, and it is what the Egg model writes;
* **enumerated arrays** - `PRESSURE`, `SWAT`, `SGAS`, `RS`, `RV`, one value per
  Cartesian cell, stating the answer directly.

`EQUIL`'s eleven-column record is read with MRST's template, whose columns nine
and eleven are not depths or pressures but *table selectors*, and whose column
ten defaults to `1` where every other column defaults to `0`. Reading the row
as "three depth/pressure pairs and five spare numbers" therefore lands the
selectors in the wrong slots; they are transcribed rather than inferred.

Values are the deck's own; `convert_deck_units` puts them in SI - which for
`EQUIL` means alternating length and pressure columns, and nothing after the
sixth.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from pymrst.modules.deckformat.deckinput.read_defaulted_record import (
    DeckCursor, read_defaulted_kw, read_defaulted_record, strip_quotes,
)

#: Per-cell arrays: one value for every Cartesian cell.
CELL_ARRAYS = ("PRESSURE", "RS", "RV", "SGAS", "SOIL", "SWAT", "TEMPI",
               "PBUB")

#: Depth tables, read as one `(depth, value)` table per equilibration region.
DEPTH_TABLES = ("PBVD", "PDVD", "RSVD", "RVVD", "RTEMPVD")

#: Read as a record and otherwise not acted on, as in MRST.
REPORT_KEYWORDS = ("RPTRST", "RPTSOL", "OUTSOL")

#: Keywords ending the section.
SECTION_KEYWORDS = ("SUMMARY", "SCHEDULE", "END", "RUNSPEC", "GRID", "EDIT",
                    "PROPS", "REGIONS")

_IGNORED = ("SOLUTION", "ECHO", "NOECHO", "ENDBOX")

# MRST's `EQUIL` template: eleven fields defaulting to zero, except the tenth.
_EQUIL_TEMPLATE = ("0",)*9 + ("1", "0")

_RECORD_TEMPLATES = {
    "AQUFETP": ("NaN",)*6 + ("1", "0", "NaN"),
    "AQUCT": ("NaN",)*8 + ("360",) + ("NaN",)*4,
    "THPRES": ("NaN", "NaN", "NaN"),
    "FIELDSEP": ("NaN", "NaN", "NaN", "0", "0", "0", "0", "0", "NaN", "NaN"),
}

#: `AQUANCON` is the one record keyword with a *text* column in the middle:
#: field 8 is a face indicator such as `'I-'`, saying which face of the box the
#: aquifer attaches to. Reading the row as all-numeric turns it into a NaN and
#: the aquifer then connects to nothing - which is silent, because a
#: disconnected analytic aquifer simply contributes no influx.
_AQUANCON_TEMPLATE = ("NaN",)*7 + ("", "NaN", "-1", "NO")
_AQUANCON_NUMERIC = tuple(range(7)) + (8, 9)


def read_solution(cursor: DeckCursor, n_cells: int,
                  solution: Optional[Dict[str, Any]] = None,
                  n_equil: int = 1) -> Dict[str, Any]:
    """Read the SOLUTION section into ``{keyword: value}``."""
    out: Dict[str, Any] = dict(solution or {})
    while True:
        keyword = cursor.next_keyword()
        if keyword is None:
            break
        if keyword in SECTION_KEYWORDS:
            cursor.rewind_keyword()     # the caller reads the next section
            break
        if keyword in _IGNORED:
            continue

        if keyword == "EQUIL":
            records = read_defaulted_kw(cursor, _EQUIL_TEMPLATE,
                                        n_rec=max(int(n_equil), 1))
            out[keyword] = np.asarray(
                [[_number(field) for field in record] for record in records],
                dtype=float)
        elif keyword == "STREQUIL":
            records = read_defaulted_kw(cursor, ("0",)*9,
                                        n_rec=max(int(n_equil), 1))
            out[keyword] = np.asarray(
                [[_number(field) for field in record] for record in records],
                dtype=float)
        elif keyword in DEPTH_TABLES:
            out[keyword] = [_table(cursor.read_record_tokens())
                            for _ in range(max(int(n_equil), 1))]
        elif keyword in CELL_ARRAYS:
            out[keyword] = _expand(cursor.read_record_tokens(), n_cells)
        elif keyword == "DATUM":
            record = cursor.read_record_tokens()
            if len(record) != 1:
                raise ValueError(
                    "DATUM must be followed by a single depth; got "
                    f"{len(record)} values")
            out[keyword] = _number(record[0])
        elif keyword == "AQUANCON":
            records = read_defaulted_kw(cursor, _AQUANCON_TEMPLATE)
            out[keyword] = [
                [(_number(field) if column in _AQUANCON_NUMERIC else str(field))
                 for column, field in enumerate(record)]
                for record in records]
        elif keyword in _RECORD_TEMPLATES:
            records = read_defaulted_kw(cursor, _RECORD_TEMPLATES[keyword])
            out[keyword] = np.asarray(
                [[_number(field) for field in record] for record in records],
                dtype=float)
        elif keyword in REPORT_KEYWORDS:
            out[keyword] = cursor.read_record_tokens()
        elif keyword == "BOX":
            cursor.read_record_tokens()
        else:
            out.setdefault("__unhandled__", [])
            if keyword not in out["__unhandled__"]:
                out["__unhandled__"].append(keyword)
    return out


readSOLUTION = read_solution


def _number(token: Any) -> float:
    if isinstance(token, (int, float)):
        return float(token)
    text = strip_quotes(str(token)).strip()
    if not text or text.lower() == "default":
        return float("nan")
    try:
        return float(text.replace("D", "e").replace("d", "e"))
    except ValueError:
        return float("nan")


def _expand(tokens: List[str], count: int) -> np.ndarray:
    """A per-cell array, honouring ECLIPSE's ``n*value`` repeat form."""
    values: List[float] = []
    for token in tokens:
        token = strip_quotes(token)
        repeat, star, value = token.partition("*")
        if star and repeat.isdigit():
            values.extend([_number(value) if value else float("nan")]
                          * int(repeat))
            continue
        values.append(_number(token))
    array = np.asarray(values, dtype=float)
    if array.size != count:
        raise ValueError(
            f"expected {count} values (one per Cartesian cell), got "
            f"{array.size}")
    return array


def _table(tokens: List[str]) -> np.ndarray:
    """A `(depth, value)` table, two columns per row."""
    values = [_number(token) for token in tokens]
    if len(values) % 2:
        raise ValueError(
            "a depth table needs an even number of values (depth, value)")
    return np.asarray(values, dtype=float).reshape(-1, 2)


__all__ = ["CELL_ARRAYS", "DEPTH_TABLES", "readSOLUTION", "read_solution"]


if __name__ == "__main__":

    def _read(text: str, n_cells: int = 4, n_equil: int = 1):
        cursor = DeckCursor.from_text(text)
        keyword = cursor.next_keyword()
        assert keyword == "SOLUTION", keyword
        return read_solution(cursor, n_cells, n_equil=n_equil)

    def test_the_egg_equil_record_reads_as_eleven_columns():
        """The Egg model's own record. Datum 4000 m at 400 bar, water-oil
        contact at 5000 m - below the reservoir, so the column is all oil."""
        solution = _read("SOLUTION\nEQUIL\n 4000 400 5000 0 /\n")
        row = solution["EQUIL"][0]
        assert row.size == 11, row.size
        assert row[0] == 4000.0 and row[1] == 400.0
        assert row[2] == 5000.0 and row[3] == 0.0
        print("ok test_the_egg_equil_record_reads_as_eleven_columns")

    def test_the_tenth_equil_column_defaults_to_one():
        """Every other column defaults to zero. Column ten is an accuracy
        selector whose zero means something different, so the asymmetry is
        real and cannot be smoothed over."""
        row = _read("SOLUTION\nEQUIL\n 4000 400 5000 0 /\n")["EQUIL"][0]
        assert row[9] == 1.0, row
        assert row[8] == 0.0 and row[10] == 0.0
        print("ok test_the_tenth_equil_column_defaults_to_one")

    def test_one_equil_record_per_region_is_read():
        """A two-region deck writes two records. Reading one leaves the second
        region equilibrated against the first region's contacts."""
        solution = _read("SOLUTION\nEQUIL\n 4000 400 5000 0 /\n"
                         " 4100 410 5100 0 /\n", n_equil=2)
        assert solution["EQUIL"].shape == (2, 11), solution["EQUIL"].shape
        assert solution["EQUIL"][1][0] == 4100.0
        print("ok test_one_equil_record_per_region_is_read")

    def test_an_all_default_equil_row_terminates_the_keyword():
        """Measured against MRST rather than assumed.

        `readDefaultedKW` compares each record with the template and stops
        when they match - and passing `NRec` does not change that, it only
        caps the count. So a second region that defaults every column ends
        the table instead of occupying a slot. Reading it as a row would give
        that region a datum depth of zero.
        """
        solution = _read("SOLUTION\nEQUIL\n 4000 400 5000 0 /\n 11* /\n",
                         n_equil=2)
        assert solution["EQUIL"].shape[0] == 1, solution["EQUIL"].shape
        print("ok test_an_all_default_equil_row_terminates_the_keyword")

    def test_enumerated_arrays_are_one_value_per_cell():
        solution = _read("SOLUTION\nPRESSURE\n 4*400 /\nSWAT\n 0.1 0.2 0.3 0.4 /\n")
        assert np.allclose(solution["PRESSURE"], 400.0)
        assert np.allclose(solution["SWAT"], [0.1, 0.2, 0.3, 0.4])
        print("ok test_enumerated_arrays_are_one_value_per_cell")

    def test_a_short_array_is_refused():
        """Padding it silently would leave the tail of the reservoir at zero
        pressure, which converges."""
        try:
            _read("SOLUTION\nPRESSURE\n 400 400 /\n")
        except ValueError as refused:
            assert "one per Cartesian cell" in str(refused)
            print("ok test_a_short_array_is_refused")
            return
        raise AssertionError("a per-cell array must cover every cell")

    def test_a_depth_table_is_two_columns():
        solution = _read("SOLUTION\nRSVD\n 4000 50 4100 45 /\n")
        table = solution["RSVD"][0]
        assert table.shape == (2, 2), table.shape
        assert table[1][1] == 45.0
        print("ok test_a_depth_table_is_two_columns")

    def test_datum_takes_exactly_one_depth():
        assert _read("SOLUTION\nDATUM\n 4000 /\n")["DATUM"] == 4000.0
        try:
            _read("SOLUTION\nDATUM\n 4000 4100 /\n")
        except ValueError:
            print("ok test_datum_takes_exactly_one_depth")
            return
        raise AssertionError("DATUM is a single depth")

    def test_report_controls_are_read_and_do_not_swallow_the_next_keyword():
        solution = _read("SOLUTION\nRPTSOL\n RESTART=2 /\n"
                         "EQUIL\n 4000 400 5000 0 /\n")
        assert solution["RPTSOL"] == ["RESTART=2"], solution["RPTSOL"]
        assert "EQUIL" in solution, "RPTSOL consumed the EQUIL record"
        print("ok test_report_controls_are_read_and_do_not_swallow_the_next_keyword")

    def test_the_section_ends_at_the_next_section_header():
        cursor = DeckCursor.from_text(
            "SOLUTION\nEQUIL\n 4000 400 5000 0 /\nSUMMARY\nFOPR\n")
        cursor.next_keyword()
        read_solution(cursor, 4)
        assert cursor.next_keyword() == "SUMMARY"
        print("ok test_the_section_ends_at_the_next_section_header")

    def test_aquancon_keeps_its_face_column_as_text():
        """Field 8 says which face of the box the aquifer attaches to. Read as
        a number it becomes NaN, the aquifer connects to nothing, and the run
        proceeds quietly with no influx at all."""
        solution = _read("SOLUTION\nAQUANCON\n 1 1 1 1 1 5 5 'J-' 1* 1 /\n/\n")
        record = solution["AQUANCON"][0]
        assert record[0] == 1.0 and record[6] == 5.0
        assert record[7] == "J-", record
        print("ok test_aquancon_keeps_its_face_column_as_text")

    def test_aqufetp_is_read_as_numbers():
        solution = _read("SOLUTION\nAQUFETP\n 1 2500 400 1e8 3e-5 /\n/\n")
        record = solution["AQUFETP"][0]
        assert record[1] == 2500.0 and record[2] == 400.0
        print("ok test_aqufetp_is_read_as_numbers")

    def test_an_unhandled_keyword_is_reported():
        solution = _read("SOLUTION\nNOSUCHKW\n 1 /\nEQUIL\n 4000 400 5000 0 /\n")
        assert solution["__unhandled__"] == ["NOSUCHKW"]
        print("ok test_an_unhandled_keyword_is_reported")

    test_the_egg_equil_record_reads_as_eleven_columns()
    test_the_tenth_equil_column_defaults_to_one()
    test_one_equil_record_per_region_is_read()
    test_an_all_default_equil_row_terminates_the_keyword()
    test_enumerated_arrays_are_one_value_per_cell()
    test_a_short_array_is_refused()
    test_a_depth_table_is_two_columns()
    test_datum_takes_exactly_one_depth()
    test_report_controls_are_read_and_do_not_swallow_the_next_keyword()
    test_the_section_ends_at_the_next_section_header()
    test_aquancon_keeps_its_face_column_as_text()
    test_aqufetp_is_read_as_numbers()
    test_an_unhandled_keyword_is_reported()
    print("\nAll tests passed!")
