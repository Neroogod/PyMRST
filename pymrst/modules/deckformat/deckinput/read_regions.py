"""The REGIONS section (MRST ``deckinput/readREGIONS.m``).

Reads the per-cell region partitions a deck declares. Each keyword is one
integer per Cartesian cell, so the section is small but load-bearing: without
it every fluid property is global, and `handle_regions` has nothing to split.

Region numbers stay **1-based** here, exactly as the deck writes them.
`handle_regions` converts to 0-based once, at the boundary between the deck
and the property layer, so there is a single place where the convention
changes rather than a conversion scattered through the readers.
"""

from typing import Dict, Optional

import numpy as np

from pymrst.modules.deckformat.deckinput.read_defaulted_record import (
    DeckCursor,
    strip_quotes,
)

# The region keywords MRST reads, all of them one integer per cell.
REGION_KEYWORDS = (
    "EQLNUM", "FIPNUM", "IMBNUM", "PVTNUM", "SATNUM",
    "SURFNUM", "ENDNUM", "ROCKNUM", "FIPFAC", "EOSNUM",
)

# Keywords that end the REGIONS section.
SECTION_KEYWORDS = ("SOLUTION", "SCHEDULE", "SUMMARY", "END", "EDIT", "PROPS")

# Recognised but carrying no data MRST keeps.
IGNORED = ("ECHO", "NOECHO")


def _expand(tokens, count: int) -> np.ndarray:
    """Expand an array record, honouring ECLIPSE's ``n*value`` repeat form."""
    values = []
    for token in tokens:
        token = strip_quotes(token)
        if "*" in token:
            repeat, _, value = token.partition("*")
            if repeat.isdigit():
                values.extend([float(value)] * int(repeat) if value
                              else [np.nan] * int(repeat))
                continue
        values.append(float(token))
    array = np.asarray(values, dtype=float)
    if array.size != count:
        raise ValueError(f"expected {count} values, got {array.size}")
    return array.astype(np.int64)


def read_regions(cursor: DeckCursor, n_cells: int,
                 regions: Optional[Dict] = None) -> Dict:
    """Read the REGIONS section, returning ``{keyword: per-cell array}``."""
    out: Dict[str, np.ndarray] = dict(regions or {})
    while True:
        keyword = cursor.next_keyword()
        if keyword is None:
            break
        if keyword in SECTION_KEYWORDS:
            cursor.rewind_keyword()     # the caller reads the next section
            break
        if keyword == "BOX":
            cursor.read_record_tokens()
            continue
        if keyword in IGNORED or keyword == "ENDBOX":
            continue
        if keyword in REGION_KEYWORDS:
            out[keyword] = _expand(cursor.read_record_tokens(), n_cells)
        # Anything else is skipped: next_keyword() steps over its data lines.
    return out


readREGIONS = read_regions


if __name__ == "__main__":
    from pymrst.modules.deckformat.deckinput.read_defaulted_record import tokenize

    def test_reads_one_value_per_cell() -> None:
        out = read_regions(DeckCursor(tokenize("SATNUM\n1 1 2 2 /\n")), 4)
        assert np.array_equal(out["SATNUM"], [1, 1, 2, 2]), out
        print("ok test_reads_one_value_per_cell")

    def test_expands_the_repeat_form() -> None:
        out = read_regions(DeckCursor(tokenize("PVTNUM\n3*1 2 /\n")), 4)
        assert np.array_equal(out["PVTNUM"], [1, 1, 1, 2]), out
        print("ok test_expands_the_repeat_form")

    def test_reads_several_region_keywords() -> None:
        text = "SATNUM\n1 2 /\nPVTNUM\n2 1 /\n"
        out = read_regions(DeckCursor(tokenize(text)), 2)
        assert set(out) == {"SATNUM", "PVTNUM"}
        assert np.array_equal(out["PVTNUM"], [2, 1])
        print("ok test_reads_several_region_keywords")

    def test_stops_at_the_next_section() -> None:
        text = "SATNUM\n1 1 /\nSOLUTION\nEQUIL\n1 2 /\n"
        cursor = DeckCursor(tokenize(text))
        out = read_regions(cursor, 2)
        assert set(out) == {"SATNUM"}
        assert cursor.peek() == "SOLUTION", cursor.peek()
        print("ok test_stops_at_the_next_section")

    def test_keeps_the_decks_one_based_numbering() -> None:
        """Conversion to 0-based happens in handle_regions, not here."""
        out = read_regions(DeckCursor(tokenize("SATNUM\n1 1 2 /\n")), 3)
        assert out["SATNUM"].min() == 1, out
        print("ok test_keeps_the_decks_one_based_numbering")

    def test_feeds_handle_regions() -> None:
        from pymrst.modules.ad_props.utils.handle_regions import handle_regions
        out = read_regions(DeckCursor(tokenize("SATNUM\n1 1 2 2 /\n")), 4)
        reg = handle_regions({"REGIONS": out, "GRID": {}})
        assert reg.count("SAT") == 2
        assert np.array_equal(reg.SATNUM, [0, 0, 1, 1]), reg.SATNUM
        print("ok test_feeds_handle_regions")

    def test_a_wrong_cell_count_is_rejected() -> None:
        try:
            read_regions(DeckCursor(tokenize("SATNUM\n1 1 /\n")), 5)
        except ValueError as error:
            assert "expected 5" in str(error)
            print("ok test_a_wrong_cell_count_is_rejected")
            return
        raise AssertionError("a short region array must be rejected")

    def test_unknown_keywords_do_not_derail_the_cursor() -> None:
        text = "MYSTERY\n1 2 3 /\nSATNUM\n1 2 /\n"
        out = read_regions(DeckCursor(tokenize(text)), 2)
        assert set(out) == {"SATNUM"}, out
        print("ok test_unknown_keywords_do_not_derail_the_cursor")

    test_reads_one_value_per_cell()
    test_expands_the_repeat_form()
    test_reads_several_region_keywords()
    test_stops_at_the_next_section()
    test_keeps_the_decks_one_based_numbering()
    test_feeds_handle_regions()
    test_a_wrong_cell_count_is_rejected()
    test_unknown_keywords_do_not_derail_the_cursor()
    print("\nAll tests passed!")
