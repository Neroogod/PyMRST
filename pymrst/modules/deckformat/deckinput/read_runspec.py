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

SECTION_KEYWORDS = ("GRID", "EDIT", "PROPS", "REGIONS",
                    "SOLUTION", "SUMMARY", "SCHEDULE", "END")

IGNORED = ("ECHO", "NOECHO")


def _floats(values) -> np.ndarray:
    return np.array([float(str(v).replace("D", "e").replace("d", "e"))
                     for v in values], dtype=float)


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
        elif keyword == "DIMENS":
            dims = _floats(cursor.read_record_tokens()[:3]).astype(np.int64)
            runspec["DIMENS"] = dims
            runspec["cartDims"] = dims.copy()
        elif keyword == "TABDIMS":
            record = read_defaulted_record(cursor, list(TABDIMS_TEMPLATE))
            runspec["TABDIMS"] = _floats(record)
        elif keyword in ("EQLDIMS", "WELLDIMS", "AQUDIMS", "REGDIMS", "COMPS"):
            runspec[keyword] = _floats(
                [strip_quotes(t) for t in cursor.read_record_tokens()
                 if not t.endswith("*")])
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

    test_reads_grid_dimensions()
    test_phase_keywords_become_flags()
    test_tabdims_gives_the_region_counts()
    test_defaulted_tabdims_falls_back_to_one_region()
    test_a_deck_without_tabdims_gets_one_region()
    test_metric_is_the_default_unit_system()
    test_a_declared_unit_system_wins()
    test_stops_at_the_next_section()
    test_dualperm_implies_dualporo()
    test_unrecognised_keywords_are_recorded_not_dropped()
    test_reads_the_spe1_runspec()
    print("\nAll tests passed!")
