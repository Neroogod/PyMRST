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

import numpy as np

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
# EDIT carries transmissibility edits this port does not apply; SUMMARY is
# output requests only, which is why MRST's own reader ignores it too.
SKIPPED_SECTIONS = ("EDIT", "SUMMARY")

INCLUDE = re.compile(r"^\s*INCLUDE\s*(?:--.*)?$", re.IGNORECASE)


def _props_array_keywords() -> Set[str]:
    """The per-cell arrays MRST's `readPROPS` reads (end points, SWATINIT,
    end-point relative permeabilities and capillary pressures)."""
    names = {"SWATINIT", "SOWCR", "SOGCR", "ISOWCR", "ISOGCR", "PCW", "IPCW",
             "PCG", "IPCG", "SWLPC", "ISWLPC", "SGLPC", "ISGLPC"}
    for stem in ("SWL", "SWCR", "SWU", "SGL", "SGCR", "SGU", "SOL", "SOCR", "SOU"):
        for d in ("", "X", "X-", "Y", "Y-", "Z", "Z-"):
            names |= {stem + d, "I" + stem + d}
    for stem in ("KRW", "KRG", "KRO", "KRORW", "KRORG"):
        for d in ("", "X", "X-", "Y", "Y-", "Z", "Z-"):
            names |= {stem + d, "I" + stem + d}
    for stem in ("KRWR", "KRGR"):
        for d in ("", "X-", "Y", "Y-", "Z", "Z-"):
            names |= {stem + d, "I" + stem + d}
        names |= {stem[:-1] + "XR", "I" + stem[:-1] + "XR"}
    return names


PROPS_ARRAY_KEYWORDS = _props_array_keywords()


def _section_texts(text: str) -> Dict[str, str]:
    """Each section's own text (a section named twice is concatenated)."""
    parts: Dict[str, List[str]] = {}
    current = None
    for line in text.splitlines():
        words = line.split("--")[0].split()
        head = words[0].upper() if words else ""
        if head in SECTIONS or head == "END":
            current = head if head in SECTIONS else None
            continue
        if current is not None:
            parts.setdefault(current, []).append(line)
    return {k: "\n".join(v) for k, v in parts.items()}


def _read_section_arrays(text: str, keywords: Set[str], sect: Dict[str, Any],
                         cart_dims, fill: float = np.nan,
                         specgrid: bool = False) -> Dict[str, Any]:
    """A section's per-cell arrays and box operators, in deck order (MRST's
    `readGridBoxArray` and `applyOperator`, inside the section only)."""
    from pymrst.modules.deckformat.deckinput.private.apply_operator import (
        OPERATORS,
        apply_operator,
        box_indices,
    )
    from pymrst.utils.inout.eclipse.deckinput_simple.read_keywords import (
        _read_array,
        _read_records,
        _skip_record,
        _tokens_from_text,
    )
    out = dict(sect)
    dims = None if cart_dims is None else [int(d) for d in cart_dims]
    box = None
    tokens = _tokens_from_text(text)
    index = 0
    while index < len(tokens):
        keyword = tokens[index].upper()
        index += 1
        if specgrid and keyword == "SPECGRID":
            dims = [int(float(tokens[index + k])) for k in range(3)]
            out["cartDims"] = tuple(dims)
            index = _skip_record(tokens, index)
        elif keyword == "BOX":
            box = [int(float(tokens[index + k])) for k in range(6)]
            index = _skip_record(tokens, index)
        elif keyword == "ENDBOX":
            box = None
        elif keyword in keywords:
            values, index = _read_array(tokens, index)
            if box is not None:
                n = int(np.prod(dims))
                array = (np.array(out[keyword], dtype=float) if keyword in out
                         else np.full(n, fill))
                array[box_indices(dims, box)] = values
                values = array
            out[keyword] = values
        elif keyword in OPERATORS:
            records, index = _read_records(tokens, index)
            if records:
                out = apply_operator(out, records, keyword, dims, box)
    return out


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
    from pymrst.modules.deckformat.deckinput.read_runspec import (
        equilibration_regions,
    )
    from pymrst.modules.deckformat.deckinput.read_schedule import read_schedule
    from pymrst.modules.deckformat.deckinput.read_solution import read_solution

    deck = initialize_deck()
    cursor = DeckCursor.from_text(text)
    has_grid = False

    while True:
        keyword = cursor.next_keyword()
        if keyword is None or keyword == "END":
            break
        if keyword == "RUNSPEC":
            read_runspec(cursor, deck)
        elif keyword == "GRID":
            # The grid arrays are read from the whole text rather than from
            # the cursor. `COPY` and `MULTIPLY` need the box dimensions that
            # `SPECGRID` states, which may sit in a different INCLUDE file
            # from the array they modify - the Egg model's own layout. Only
            # grid keywords are recognised, so scanning past the section
            # boundary picks nothing up from PROPS or SOLUTION.
            has_grid = True
        elif keyword == "PROPS":
            deck["PROPS"] = read_props(cursor, table_counts(deck),
                                       deck["PROPS"])
        elif keyword == "REGIONS":
            n_cells = int(deck["RUNSPEC"]["cartDims"].prod())
            deck["REGIONS"] = read_regions(cursor, n_cells, deck["REGIONS"])
        elif keyword == "SOLUTION":
            n_cells = int(deck["RUNSPEC"]["cartDims"].prod())
            deck["SOLUTION"] = read_solution(
                cursor, n_cells, deck["SOLUTION"],
                n_equil=equilibration_regions(deck))
            unhandled = deck["SOLUTION"].pop("__unhandled__", [])
            deck["UnhandledKeywords"]["SOLUTION"].extend(unhandled)
        elif keyword == "SCHEDULE":
            read_schedule(cursor, deck)
        elif keyword in SKIPPED_SECTIONS:
            # Recognised as a section boundary and stepped over. next_keyword()
            # walks the section's data, and the loop resumes at what follows.
            deck["UnhandledKeywords"][keyword].append(keyword)
        # Anything else at this level is not a section; skip it.

    sections = _section_texts(text)
    dims = deck["RUNSPEC"].get("cartDims")
    if has_grid:
        from pymrst.utils.inout.eclipse.deckinput_simple.read_keywords import (
            ARRAY_KEYWORDS,
        )
        deck["GRID"] = _read_section_arrays(sections.get("GRID", ""), set(ARRAY_KEYWORDS),
                                            {}, dims, specgrid=True)
    if "PROPS" in sections and dims is not None:
        deck["PROPS"] = _read_section_arrays(sections["PROPS"], PROPS_ARRAY_KEYWORDS,
                                             deck["PROPS"], dims)
    if "REGIONS" in sections and dims is not None:
        from pymrst.modules.deckformat.deckinput.read_regions import REGION_KEYWORDS
        regions = _read_section_arrays(sections["REGIONS"], set(REGION_KEYWORDS), {},
                                       dims, fill=1.0)
        deck["REGIONS"] = {k: (np.asarray(v).astype(np.int64)
                               if not np.any(np.isnan(np.asarray(v, dtype=float))) else v)
                           for k, v in regions.items()}
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

    def test_operators_act_in_their_own_section() -> None:
        """Norne sets SATNUM/IMBNUM with EQUALS in REGIONS and builds its end
        points with COPY/ADD in PROPS; MRST keeps each array in its section."""
        text = ("RUNSPEC\nDIMENS\n2 1 1 /\n"
                "GRID\nPORO\n2*0.3 /\n"
                "PROPS\nSWL\n0.1 0.2 /\nCOPY\nSWL SWCR /\n/\n"
                "ADD\nSWCR 0.05 1 1 1 1 1 1 /\n/\nSCALECRS\nYES /\n"
                "REGIONS\nEQUALS\n'SATNUM' 1 /\n'IMBNUM' 2 /\n/\nEND\n")
        deck = read_eclipse_deck_text(text)
        assert set(deck["GRID"]) == {"PORO", "cartDims"}, set(deck["GRID"])
        np.testing.assert_allclose(deck["PROPS"]["SWCR"], [0.15, 0.2])
        assert deck["PROPS"]["SCALECRS"] == ["YES"]
        np.testing.assert_array_equal(deck["REGIONS"]["IMBNUM"], [2, 2])
        np.testing.assert_array_equal(deck["REGIONS"]["SATNUM"], [1, 1])
        print("ok test_operators_act_in_their_own_section")

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

    def test_the_grid_section_reaches_the_deck() -> None:
        """GRID used to be listed among the skipped sections, so a deck's
        petrophysics never arrived and a caller had to read it separately."""
        text = ("RUNSPEC\nDIMENS\n1 1 1 /\n"
                "GRID\nPORO\n0.3 /\nPERMX\n100 /\n"
                "PROPS\nDENSITY\n900 1000 0.8 /\n")
        deck = read_eclipse_deck_text(text)
        assert np.allclose(deck["GRID"]["PORO"], 0.3)
        assert np.allclose(deck["GRID"]["PERMX"], 100.0)
        assert "DENSITY" in deck["PROPS"], "PROPS must survive the GRID reader"
        print("ok test_the_grid_section_reaches_the_deck")

    def test_the_solution_and_schedule_sections_reach_the_deck() -> None:
        text = ("RUNSPEC\nDIMENS\n1 1 1 /\nSTART\n15 JUN 2011 /\n"
                "SOLUTION\nEQUIL\n4000 400 5000 0 /\n"
                "SCHEDULE\n"
                "WELSPECS\n 'P1' 'G' 1 1 1* 'OIL' /\n/\n"
                "COMPDAT\n 'P1' 2* 1 1 'OPEN' /\n/\n"
                "WCONPROD\n 'P1' 'OPEN' 'BHP' 5* 395 /\n/\n"
                "TSTEP\n 3*30 /\nEND\n")
        deck = read_eclipse_deck_text(text)
        assert deck["SOLUTION"]["EQUIL"][0][1] == 400.0
        assert deck["RUNSPEC"]["START"].year == 2011
        schedule = deck["SCHEDULE"]
        assert len(schedule.control) == 1, len(schedule.control)
        assert schedule.step["val"] == [30.0]*3
        assert len(schedule.control[0]["WELSPECS"]) == 1
        print("ok test_the_solution_and_schedule_sections_reach_the_deck")

    def test_an_unported_section_is_skipped_not_fatal() -> None:
        """EDIT and SUMMARY remain skipped: output requests and
        transmissibility edits this port does not apply."""
        text = ("RUNSPEC\nDIMENS\n1 1 1 /\n"
                "SUMMARY\nFOPR\n"
                "SCHEDULE\nTSTEP\n 10 /\n")
        deck = read_eclipse_deck_text(text)
        assert "SUMMARY" in deck["UnhandledKeywords"]["SUMMARY"]
        assert deck["SCHEDULE"].step["val"] == [10.0]
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

    test_operators_act_in_their_own_section()
    test_reads_sections_in_order()
    test_tabdims_reaches_the_props_reader()
    test_the_grid_section_reaches_the_deck()
    test_the_solution_and_schedule_sections_reach_the_deck()
    test_an_unported_section_is_skipped_not_fatal()
    test_end_stops_the_reader()
    test_include_is_spliced_in()
    test_a_missing_include_is_reported()
    test_reads_the_spe1_deck_end_to_end()
    print("\nAll tests passed!")
