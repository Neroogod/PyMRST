"""Read an ECLIPSE deck (MRST ``deckinput/readEclipseDeck.m``).

MRST chains its section readers recursively - `readRUNSPEC` ends by calling
`readGRID`, which calls `readPROPS`, and so on. The readers here instead stop
at a section boundary and hand the cursor back, so the sequencing lives in one
flat loop below. The observable result is the same deck; the difference is
that a section this port has not yet covered can be skipped without breaking
the chain, which is what lets `ad_props` use the PROPS reader before the GRID
and SCHEDULE readers exist.

INCLUDE files are spliced in before tokenizing. Doing it on the text means an
included file behaves exactly as if it had been pasted in, including for the
"a keyword starts a line" rule that the whole reader depends on.
"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from pymrst.modules.deckformat.deckinput.initialize_deck import initialize_deck
from pymrst.modules.deckformat.deckinput.read_defaulted_record import DeckCursor
from pymrst.modules.deckformat.deckinput.read_props import read_props
from pymrst.modules.deckformat.deckinput.read_regions import read_regions
from pymrst.modules.deckformat.deckinput.read_runspec import (
    read_runspec,
    table_counts,
)

# Sections this reader knows how to parse. The rest are recognised as section
# boundaries and skipped - see `SKIPPED_SECTIONS`.
SECTIONS = ("RUNSPEC", "GRID", "EDIT", "PROPS", "REGIONS",
            "SOLUTION", "SUMMARY", "SCHEDULE")
SKIPPED_SECTIONS = ("GRID", "EDIT", "SOLUTION", "SUMMARY", "SCHEDULE")

INCLUDE = re.compile(r"^\s*INCLUDE\s*(?:--.*)?$", re.IGNORECASE)


def expand_includes(text: str, directory: Path,
                    seen: Optional[Set[Path]] = None) -> str:
    """Splice INCLUDE'd files into ``text``, recursively."""
    seen = seen if seen is not None else set()
    out: List[str] = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        if not INCLUDE.match(line.split("--")[0]):
            out.append(line)
            index += 1
            continue
        index += 1
        # The filename is the next non-blank, non-comment line, quoted and
        # terminated by a slash.
        while index < len(lines) and not lines[index].split("--")[0].strip():
            index += 1
        if index >= len(lines):
            break
        record = lines[index].split("--")[0].strip().rstrip("/").strip()
        index += 1
        name = record.strip().strip("'\"").strip()
        path = (directory / name).resolve()
        if path in seen:
            raise ValueError(f"INCLUDE cycle through {path}")
        if not path.exists():
            raise FileNotFoundError(f"INCLUDE file not found: {path}")
        out.append(f"-- INCLUDE {name}")
        out.append(expand_includes(path.read_text(errors="ignore"),
                                   path.parent, seen | {path}))
    return "\n".join(out)


def read_eclipse_deck(path) -> Dict[str, Any]:
    """Read a deck from ``path``, returning MRST's deck structure."""
    path = Path(path)
    text = expand_includes(path.read_text(errors="ignore"), path.parent)
    return read_eclipse_deck_text(text)


def read_eclipse_deck_text(text: str) -> Dict[str, Any]:
    """Read a deck already assembled into a single string."""
    deck = initialize_deck()
    cursor = DeckCursor.from_text(text)

    while True:
        keyword = cursor.next_keyword()
        if keyword is None or keyword == "END":
            break
        if keyword == "RUNSPEC":
            read_runspec(cursor, deck)
        elif keyword == "PROPS":
            deck["PROPS"] = read_props(cursor, table_counts(deck),
                                       deck["PROPS"])
        elif keyword == "REGIONS":
            n_cells = int(deck["RUNSPEC"]["cartDims"].prod())
            deck["REGIONS"] = read_regions(cursor, n_cells, deck["REGIONS"])
        elif keyword in SKIPPED_SECTIONS:
            # Not yet ported. next_keyword() steps over the section's data,
            # and the loop resumes at whatever section follows.
            deck["UnhandledKeywords"][keyword].append(keyword)
        # Anything else at this level is not a section; skip it.

    if deck["GRID"] is not None and deck["RUNSPEC"].get("cartDims") is not None:
        # Convenience for the rest of MRST, as in readEclipseDeck.m.
        deck["GRID"]["cartDims"] = deck["RUNSPEC"]["cartDims"]
    return deck


readEclipseDeck = read_eclipse_deck


if __name__ == "__main__":
    import tempfile

    import numpy as np

    DATA = Path("/Users/Shared/PyMRST_workspace/"
                "mrst-2024a-Geomech/examples/data")

    def test_reads_sections_in_order() -> None:
        text = ("RUNSPEC\nDIMENS\n2 1 1 /\nOIL\n"
                "PROPS\nDENSITY\n900 1000 0.8 /\n"
                "REGIONS\nSATNUM\n1 2 /\nEND\n")
        deck = read_eclipse_deck_text(text)
        assert np.array_equal(deck["RUNSPEC"]["cartDims"], [2, 1, 1])
        assert "DENSITY" in deck["PROPS"]
        assert np.array_equal(deck["REGIONS"]["SATNUM"], [1, 2])
        print("ok test_reads_sections_in_order")

    def test_tabdims_reaches_the_props_reader() -> None:
        """RUNSPEC must be read first, or a two-region deck loses a table."""
        text = ("RUNSPEC\nDIMENS\n1 1 1 /\nTABDIMS\n2 1 /\n"
                "PROPS\nSWOF\n0.1 0.0 1.0 0.0 /\n0.2 0.1 0.9 0.0 /\n")
        deck = read_eclipse_deck_text(text)
        assert len(deck["PROPS"]["SWOF"]) == 2, len(deck["PROPS"]["SWOF"])
        print("ok test_tabdims_reaches_the_props_reader")

    def test_an_unported_section_is_skipped_not_fatal() -> None:
        text = ("RUNSPEC\nDIMENS\n1 1 1 /\n"
                "GRID\nPORO\n0.3 /\n"
                "PROPS\nDENSITY\n900 1000 0.8 /\n")
        deck = read_eclipse_deck_text(text)
        assert "DENSITY" in deck["PROPS"], "PROPS must survive a skipped GRID"
        assert "GRID" in deck["UnhandledKeywords"]["GRID"]
        print("ok test_an_unported_section_is_skipped_not_fatal")

    def test_end_stops_the_reader() -> None:
        text = ("RUNSPEC\nDIMENS\n1 1 1 /\nEND\n"
                "PROPS\nDENSITY\n900 1000 0.8 /\n")
        assert read_eclipse_deck_text(text)["PROPS"] == {}
        print("ok test_end_stops_the_reader")

    def test_include_is_spliced_in() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "props.inc").write_text("DENSITY\n900 1000 0.8 /\n")
            (root / "deck.DATA").write_text(
                "RUNSPEC\nDIMENS\n1 1 1 /\nPROPS\nINCLUDE\n 'props.inc' /\n")
            deck = read_eclipse_deck(root / "deck.DATA")
        assert np.allclose(deck["PROPS"]["DENSITY"][0], [900, 1000, 0.8])
        print("ok test_include_is_spliced_in")

    def test_a_missing_include_is_reported() -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "deck.DATA").write_text("PROPS\nINCLUDE\n 'nope.inc' /\n")
            try:
                read_eclipse_deck(root / "deck.DATA")
            except FileNotFoundError as error:
                assert "nope.inc" in str(error)
                print("ok test_a_missing_include_is_reported")
                return
        raise AssertionError("a missing INCLUDE must be reported")

    def test_reads_the_spe1_deck_end_to_end() -> None:
        path = DATA / "SPE1" / "BENCH_SPE1.DATA"
        if not path.exists():
            print("skip test_reads_the_spe1_deck_end_to_end (no MRST data)")
            return
        deck = read_eclipse_deck(path)
        assert np.array_equal(deck["RUNSPEC"]["cartDims"], [10, 10, 3])
        assert deck["RUNSPEC"].get("FIELD") is True
        assert deck["PROPS"]["SWOF"][0].shape == (14, 4)
        assert deck["PROPS"]["PVTO"][0].key.size == 9
        print("ok test_reads_the_spe1_deck_end_to_end")

    test_reads_sections_in_order()
    test_tabdims_reaches_the_props_reader()
    test_an_unported_section_is_skipped_not_fatal()
    test_end_stops_the_reader()
    test_include_is_spliced_in()
    test_a_missing_include_is_reported()
    test_reads_the_spe1_deck_end_to_end()
    print("\nAll tests passed!")
