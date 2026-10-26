"""The RUNSPEC section (MRST ``deckinput/readRUNSPEC.m``).

RUNSPEC declares the shape of everything that follows: the grid dimensions,
which phases are present, the unit system, and - the reason `ad_props` needs
it - TABDIMS, which says how many saturation and PVT regions the PROPS tables
will contain. A PROPS reader that guesses one region reads the first table of
a multi-region deck and silently drops the rest, so TABDIMS has to be read
before PROPS, not after.

Coverage here is the part of RUNSPEC that PROPS and REGIONS depend on. Other
RUNSPEC keywords are recorded as unhandled rather than dropped, so a caller
can see what a deck asked for.
"""

from typing import Any, Dict, List, Optional

import numpy as np

from pymrst.modules.deckformat.deckinput.read_defaulted_record import (
    DeckCursor,
    read_defaulted_record,
    strip_quotes,
)

# Keywords whose presence is the whole statement.
FLAGS = (
    "NOGRAV", "IMPES", "METRIC", "FIELD", "LAB", "PVT-M",
    "WATER", "OIL", "GAS", "SOLVENT", "DISGAS", "VAPOIL", "BLACKOIL",
    "POLYMER", "SURFACT", "BRINE", "TEMP", "THERMAL", "MECH",
)

UNIT_SYSTEMS = ("METRIC", "FIELD", "LAB", "PVT-M")

# TABDIMS defaults, transcribed from readRUNSPEC.m. Only the first eight
# matter downstream, but the tail is kept so field 10 (NTFIP) and friends land
# in the right slot when a deck defaults its way past them.
TABDIMS_TEMPLATE = ("1", "1", "20", "20", "1", "20", "20", "1",
                    "1", "NaN", "10", "1", "-1", "0", "0", "NaN",
                    "10", "10", "10", "NaN", "5", "5", "5", "0", "NaN")

# The remaining dimension keywords, likewise transcribed. A template is what
# makes `2* 100 2*` mean "fields one and two take their defaults": reading only
# the values a deck states shifts 100 into the first slot, and NTEQUL - the
# number of equilibration regions - then reads as 100 instead of 1.
DIMENSION_TEMPLATES = {
    "EQLDIMS": ("1", "100", "50", "1", "50"),
    "WELLDIMS": ("0", "0", "0", "0", "5", "10", "5",
                 "4", "3", "0", "1", "1", "10", "201"),
    "AQUDIMS": ("1", "1", "1", "36", "1", "1", "0", "0"),
    "REGDIMS": ("1", "1", "0", "0", "0", "1", "0", "0", "0", "0"),
}

_MONTHS = {"JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
           "JLY": 7, "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11,
           "DEC": 12}

SECTION_KEYWORDS = ("GRID", "EDIT", "PROPS", "REGIONS",
                    "SOLUTION", "SUMMARY", "SCHEDULE", "END")

IGNORED = ("ECHO", "NOECHO")


def _floats(values) -> np.ndarray:
    return np.array([float(str(v).replace("D", "e").replace("d", "e"))
                     for v in values], dtype=float)


def _read_text_flags(cursor: DeckCursor, opt: Dict[str, bool], kw: str) -> Dict[str, bool]:
    """MRST's `readTextFlags`: each name in the record sets its flag."""
    import warnings
    opt = dict(opt)
    for flag in (strip_quotes(t) for t in cursor.read_record_tokens()):
        if not flag.isidentifier():
            continue
        if flag in opt:
            opt[flag] = True
        else:
            warnings.warn(f"Option '{flag}' is not supported in keyword '{kw}'")
    return opt


def read_runspec(cursor: DeckCursor,
                 deck: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Read RUNSPEC into ``deck['RUNSPEC']``, stopping at the next section."""
    if deck is None:
        from pymrst.modules.deckformat.deckinput.initialize_deck import (
            initialize_deck,
        )
        deck = initialize_deck()
    runspec = deck["RUNSPEC"]
    unhandled: List[str] = deck["UnhandledKeywords"]["RUNSPEC"]

    while True:
        keyword = cursor.next_keyword()
        if keyword is None:
            break
        if keyword in SECTION_KEYWORDS:
            cursor.rewind_keyword()
            break
        if keyword in IGNORED:
            continue

        if keyword in FLAGS:
            runspec[keyword] = True
        elif keyword == "ENDSCALE":
            # MRST: { 'NODIR', 'REVERS', '1', '20', '0' }, the last three numeric.
            record = read_defaulted_record(cursor, ["NODIR", "REVERS", "1", "20", "0"])
            runspec["ENDSCALE"] = list(record[:2]) + [float(v) for v in record[2:]]
        elif keyword == "EQLOPTS":
            # MRST: the record's words, unquoted.
            runspec["EQLOPTS"] = [strip_quotes(t) for t in cursor.read_record_tokens()]
        elif keyword == "SATOPTS":
            runspec["SATOPTS"] = _read_text_flags(cursor, runspec["SATOPTS"], keyword)
        elif keyword == "DIMENS":
            dims = _floats(cursor.read_record_tokens()[:3]).astype(np.int64)
            runspec["DIMENS"] = dims
            runspec["cartDims"] = dims.copy()
        elif keyword == "TABDIMS":
            record = read_defaulted_record(cursor, list(TABDIMS_TEMPLATE))
            runspec["TABDIMS"] = _floats(record)
        elif keyword in DIMENSION_TEMPLATES:
            record = read_defaulted_record(cursor,
                                           list(DIMENSION_TEMPLATES[keyword]))
            runspec[keyword] = _floats(record)
        elif keyword == "COMPS":
            runspec[keyword] = _floats(
                [strip_quotes(t) for t in cursor.read_record_tokens()
                 if not t.endswith("*")])
        elif keyword == "TITLE":
            runspec["TITLE"] = _title(cursor)
        elif keyword == "START":
            runspec["START"] = _start_date(cursor.read_record_tokens())
        elif keyword in ("DUALPORO", "DUALPERM", "NODPPM"):
            runspec[keyword] = True
            if keyword == "DUALPERM":
                runspec["DUALPORO"] = True
        else:
            unhandled.append(keyword)

    if not any(runspec.get(system) for system in UNIT_SYSTEMS):
        # "Unit system not specified in input deck. Defaulting to 'METRIC'
        # conventions." - readEclipseDeck.m, following the ECLIPSE spec.
        runspec["METRIC"] = True
    return deck


def _title(cursor: DeckCursor) -> str:
    """MRST's `strtrim(fgetl(fid))`: the whole next line, as free text."""
    if not cursor:
        return ""
    if cursor.lines is None:
        return strip_quotes(cursor.next())
    line = cursor.lines[cursor.index]
    words = []
    while cursor and cursor.lines[cursor.index] == line:
        words.append(cursor.next())
    return " ".join(words).strip()


def _start_date(record: List[str]):
    """`15 'JUN' 2011` - the day/month-name/year form ECLIPSE writes.

    Left unread, the start date stays at ECLIPSE's 1983-01-01 default, and
    every `DATES` step in the SCHEDULE is then measured from the wrong origin.
    """
    from datetime import date

    fields = [strip_quotes(str(field)).strip() for field in record if field]
    if len(fields) < 3:
        raise ValueError(f"START record {record!r} is not a day, month, year")
    month = fields[1].upper()[:3]
    if month not in _MONTHS:
        raise ValueError(f"unknown month {fields[1]!r} in START")
    return date(int(float(fields[2])), _MONTHS[month], int(float(fields[0])))


def equilibration_regions(deck: Dict[str, Any]) -> int:
    """NTEQUL: how many equilibration regions the SOLUTION section declares."""
    eqldims = deck["RUNSPEC"].get("EQLDIMS")
    if eqldims is None or not np.size(eqldims):
        return 1
    count = float(np.ravel(eqldims)[0])
    return int(count) if np.isfinite(count) and count >= 1 else 1


def table_counts(deck: Dict[str, Any]):
    """The region counts PROPS needs, read off TABDIMS."""
    from pymrst.modules.deckformat.deckinput.read_props import TableCounts

    tabdims = deck["RUNSPEC"].get("TABDIMS")
    if tabdims is None:
        return TableCounts()
    return TableCounts(
        ntsfun=int(tabdims[0]),          # NTSFUN: saturation function tables
        ntpvt=int(tabdims[1]),           # NTPVT:  PVT tables
        ntrocc=int(tabdims[12]) if tabdims[12] > 0 else 1,   # NTROCC
        ntmisc=int(tabdims[8]),          # NTMISC (miscibility)
    )


def unit_system(deck: Dict[str, Any]) -> str:
    """Which of METRIC, FIELD, LAB or PVT-M the deck declared."""
    for system in UNIT_SYSTEMS:
        if deck["RUNSPEC"].get(system):
            return system
    return "METRIC"


readRUNSPEC = read_runspec


if __name__ == "__main__":

    def test_the_title_is_the_whole_next_line() -> None:
        deck = read_runspec(DeckCursor.from_text(
            "TITLE\n   ODEH PROBLEM - IMPLICIT OPTION - 1200 DAYS\nOIL\n"))
        assert deck["RUNSPEC"]["TITLE"] == "ODEH PROBLEM - IMPLICIT OPTION - 1200 DAYS"
        assert deck["RUNSPEC"]["OIL"] is True
        print("ok test_the_title_is_the_whole_next_line")

    def test_reads_grid_dimensions() -> None:
        deck = read_runspec(DeckCursor.from_text("DIMENS\n10 20 3 /\n"))
        assert np.array_equal(deck["RUNSPEC"]["cartDims"], [10, 20, 3])
        print("ok test_reads_grid_dimensions")

    def test_phase_keywords_become_flags() -> None:
        deck = read_runspec(DeckCursor.from_text("OIL\nGAS\nWATER\nDISGAS\n"))
        runspec = deck["RUNSPEC"]
        assert runspec["OIL"] and runspec["GAS"] and runspec["DISGAS"]
        print("ok test_phase_keywords_become_flags")

    def test_tabdims_gives_the_region_counts() -> None:
        deck = read_runspec(DeckCursor.from_text("TABDIMS\n3 2 30 30 /\n"))
        counts = table_counts(deck)
        assert counts.ntsfun == 3 and counts.ntpvt == 2, counts
        print("ok test_tabdims_gives_the_region_counts")

    def test_defaulted_tabdims_falls_back_to_one_region() -> None:
        deck = read_runspec(DeckCursor.from_text("TABDIMS\n2* 30 /\n"))
        counts = table_counts(deck)
        assert counts.ntsfun == 1 and counts.ntpvt == 1, counts
        print("ok test_defaulted_tabdims_falls_back_to_one_region")

    def test_a_deck_without_tabdims_gets_one_region() -> None:
        counts = table_counts(read_runspec(DeckCursor.from_text("OIL\n")))
        assert counts.ntsfun == 1 and counts.ntpvt == 1
        print("ok test_a_deck_without_tabdims_gets_one_region")

    def test_metric_is_the_default_unit_system() -> None:
        deck = read_runspec(DeckCursor.from_text("OIL\n"))
        assert unit_system(deck) == "METRIC"
        assert deck["RUNSPEC"]["METRIC"] is True
        print("ok test_metric_is_the_default_unit_system")

    def test_a_declared_unit_system_wins() -> None:
        deck = read_runspec(DeckCursor.from_text("FIELD\nOIL\n"))
        assert unit_system(deck) == "FIELD"
        assert "METRIC" not in deck["RUNSPEC"]
        print("ok test_a_declared_unit_system_wins")

    def test_stops_at_the_next_section() -> None:
        cursor = DeckCursor.from_text("OIL\nGRID\nDIMENS\n1 1 1 /\n")
        read_runspec(cursor)
        assert cursor.next_keyword() == "GRID"
        print("ok test_stops_at_the_next_section")

    def test_dualperm_implies_dualporo() -> None:
        deck = read_runspec(DeckCursor.from_text("DUALPERM\n"))
        assert deck["RUNSPEC"]["DUALPORO"] is True
        print("ok test_dualperm_implies_dualporo")

    def test_unrecognised_keywords_are_recorded_not_dropped() -> None:
        deck = read_runspec(DeckCursor.from_text("OIL\nMYSTERY\n1 2 /\n"))
        assert "MYSTERY" in deck["UnhandledKeywords"]["RUNSPEC"]
        print("ok test_unrecognised_keywords_are_recorded_not_dropped")

    def test_the_start_date_is_read() -> None:
        """Left unread it stays at ECLIPSE's 1983-01-01 default, and every
        `DATES` step in the SCHEDULE is then measured from the wrong origin."""
        from datetime import date
        deck = read_runspec(DeckCursor.from_text("START\n15 'JUN' 2011 /\n"))
        assert deck["RUNSPEC"]["START"] == date(2011, 6, 15)
        print("ok test_the_start_date_is_read")

    def test_a_defaulted_eqldims_gives_one_equilibration_region() -> None:
        """The Egg model writes `2* 100 2*`. Reading only the stated values
        shifts 100 into the first slot, and the deck then claims a hundred
        equilibration regions where it has one - so the SOLUTION reader looks
        for a hundred EQUIL records."""
        deck = read_runspec(DeckCursor.from_text("EQLDIMS\n2* 100 2* /\n"))
        assert equilibration_regions(deck) == 1, deck["RUNSPEC"]["EQLDIMS"]
        assert deck["RUNSPEC"]["EQLDIMS"][2] == 100.0
        print("ok test_a_defaulted_eqldims_gives_one_equilibration_region")

    def test_an_explicit_eqldims_count_is_kept() -> None:
        deck = read_runspec(DeckCursor.from_text("EQLDIMS\n3 100 50 /\n"))
        assert equilibration_regions(deck) == 3
        print("ok test_an_explicit_eqldims_count_is_kept")

    def test_reads_the_spe1_runspec() -> None:
        from pathlib import Path
        path = Path("/Users/Shared/PyMRST_workspace/"
                    "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA")
        if not path.exists():
            print("skip test_reads_the_spe1_runspec (no MRST data)")
            return
        cursor = DeckCursor.from_text(path.read_text(errors="ignore"))
        while cursor.next_keyword() != "RUNSPEC":
            pass
        deck = read_runspec(cursor)
        assert np.array_equal(deck["RUNSPEC"]["cartDims"], [10, 10, 3]), \
            deck["RUNSPEC"]["cartDims"]
        assert unit_system(deck) == "FIELD", unit_system(deck)
        assert deck["RUNSPEC"].get("DISGAS") is True
        print("ok test_reads_the_spe1_runspec")

    test_the_title_is_the_whole_next_line()
    test_reads_grid_dimensions()
    test_phase_keywords_become_flags()
    test_tabdims_gives_the_region_counts()
    test_defaulted_tabdims_falls_back_to_one_region()
    test_a_deck_without_tabdims_gets_one_region()
    def test_satopts_sets_its_flags() -> None:
        deck = read_runspec(DeckCursor.from_text("SATOPTS\nHYSTER  /\n"))
        assert deck["RUNSPEC"]["SATOPTS"] == {"DIRECT": False, "HYSTER": True,
                                              "IRREVERS": False}
        print("ok test_satopts_sets_its_flags")

    def test_eqlopts_keeps_its_words() -> None:
        deck = read_runspec(DeckCursor.from_text("EQLOPTS\n 'THPRES'  /  a comment\n"))
        assert deck["RUNSPEC"]["EQLOPTS"] == ["THPRES"], deck["RUNSPEC"]["EQLOPTS"]
        print("ok test_eqlopts_keeps_its_words")

    def test_endscale_keeps_mrsts_defaults() -> None:
        deck = read_runspec(DeckCursor.from_text("ENDSCALE\nNODIR   REVERS /\n"))
        assert deck["RUNSPEC"]["ENDSCALE"] == ["NODIR", "REVERS", 1.0, 20.0, 0.0]
        print("ok test_endscale_keeps_mrsts_defaults")

    test_endscale_keeps_mrsts_defaults()
    test_eqlopts_keeps_its_words()
    test_satopts_sets_its_flags()
    test_metric_is_the_default_unit_system()
    test_a_declared_unit_system_wins()
    test_stops_at_the_next_section()
    test_dualperm_implies_dualporo()
    test_unrecognised_keywords_are_recorded_not_dropped()
    test_the_start_date_is_read()
    test_a_defaulted_eqldims_gives_one_equilibration_region()
    test_an_explicit_eqldims_count_is_kept()
    test_reads_the_spe1_runspec()
    print("\nAll tests passed!")
