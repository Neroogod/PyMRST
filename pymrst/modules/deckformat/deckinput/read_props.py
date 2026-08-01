"""The PROPS section (MRST ``deckinput/readPROPS.m``).

Reads the fluid-property keywords a deck declares. MRST writes this as a 400
line `switch`; the shape underneath is highly regular - almost every keyword is
one of a handful of reader forms - so it is expressed here as a dispatch
table. Adding a keyword is a table entry, not a new branch, which is what
makes the remaining coverage tractable.

The forms, matching MRST's helpers:

- ``rel_perm``: saturation-function tables, one per saturation region;
- ``immiscible``: plain PVT tables, one per PVT region;
- ``miscible``: PVTO/PVTG, returned packed by key;
- ``record``: a fixed template read once per region;
- ``vector``: a flat list, used by the compositional keywords;
- ``mixed_record``: a record whose fields are partly numbers and partly mode
  words, so it stays a list rather than becoming an array (EHYSTR only);
- ``composite``: a record *and* a table under one keyword (PLYSHLOG only).

Values come back exactly as written in the deck. Unit conversion is
`convert_deck_units`' job, so a reader never has to know which unit system the
deck declared.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from pymrst.modules.deckformat.deckinput.read_defaulted_record import (
    DeckCursor,
    read_defaulted_kw,
    read_defaulted_record,
    read_fixed_num_records,
    strip_quotes,
)
from pymrst.modules.deckformat.deckinput.read_pvt_tables import (
    read_immiscible_pvt_table,
    read_miscible_pvt_table,
)
from pymrst.modules.deckformat.deckinput.read_rel_perm_table import (
    read_rel_perm_table,
)

SECTION_KEYWORDS = ("REGIONS", "SOLUTION", "SCHEDULE", "SUMMARY", "END", "EDIT")
IGNORED = ("ECHO", "NOECHO", "SCALECRS", "RKTRMDIR", "PRCORR",
           "STONE", "STONE1", "STONE2", "SIMPLE")


@dataclass(frozen=True)
class Spec:
    """How one keyword is read: which form, how many tables, how wide."""

    form: str
    columns: Optional[int] = None
    regions: str = "ntpvt"
    template: Optional[tuple] = None
    # readFixedNumRecords rather than readDefaultedKW: an all-default record
    # is data, not a terminator.
    fixed: bool = False
    # For a mixed record, which fields are numbers rather than mode words.
    numeric: tuple = ()


# Transcribed from MRST's switch, keyword by keyword.
SPECS: Dict[str, Spec] = {
    # saturation functions
    "SWOF": Spec("rel_perm", 4, "ntsfun"),
    "SGOF": Spec("rel_perm", 4, "ntsfun"),
    "SLGOF": Spec("rel_perm", 4, "ntsfun"),
    "SGWFN": Spec("rel_perm", 4, "ntsfun"),
    "SWFN": Spec("rel_perm", 3, "ntsfun"),
    "SGFN": Spec("rel_perm", 3, "ntsfun"),
    "SOF3": Spec("rel_perm", 3, "ntsfun"),
    "SSFN": Spec("rel_perm", 3, "ntsfun"),
    "SOF2": Spec("rel_perm", 2, "ntsfun"),
    "MSFN": Spec("rel_perm", 3, "ntsfun"),
    # PVT
    "PVDO": Spec("immiscible", 3), "PVDG": Spec("immiscible", 3),
    "PVDS": Spec("immiscible", 3), "PVCO": Spec("immiscible", 6),
    "PVTO": Spec("miscible", 3), "PVTG": Spec("miscible", 3),
    # Record templates are transcribed field-for-field from readPROPS.m: the
    # width sets how many columns a defaulted record produces, and the default
    # itself distinguishes "absent" (NaN) from "zero", which downstream code
    # relies on - SHRATE and STCOND both substitute a value for NaN.
    "PVTW": Spec("record", template=("0.0",) * 5, fixed=True),
    "PVCDO": Spec("record", template=("0.0",) * 5),
    "DENSITY": Spec("record", template=("0.0",) * 3),
    "SDENSITY": Spec("record", template=("0.0",)),
    "GRAVITY": Spec("record", template=("45.5", "1.0", "0.7773")),
    "RSCONSTT": Spec("record", template=("-1.0", "-1.0")),
    "ROCK": Spec("record", template=("NaN",) * 6, fixed=True),
    "ROCKTAB": Spec("rel_perm", 3, "ntrocc"),
    "RTEMP": Spec("record", template=("NaN",), regions="one"),
    "RTEMPA": Spec("record", template=("NaN",), regions="one"),
    "STCOND": Spec("record", template=("NaN",) * 2, regions="one"),
    # thermal
    "SPECHEAT": Spec("rel_perm", 4, "ntsfun"),
    "SPECROCK": Spec("rel_perm", 2, "ntsfun"),
    # hysteresis: one record mixing numbers and mode words, so it stays a list
    # of fields rather than becoming an array. Fields 1-4 and 12-13 are
    # numeric; the rest name modes.
    "EHYSTR": Spec("mixed_record",
                   template=("0.1", "2", "1.0", "0.1",
                             "BOTH", "RETR", "DRAIN", "OIL",
                             "NO", "NO", "NO", "0.0", "0"),
                   numeric=(0, 1, 2, 3, 11, 12)),
    # polymer
    "PLYADS": Spec("rel_perm", 2, "ntsfun"),
    "PLYVISC": Spec("immiscible", 2), "PLYSHEAR": Spec("immiscible", 2),
    "PLYMAX": Spec("record", template=("0.0",) * 2, regions="ntmisc"),
    "PLYROCK": Spec("record", template=("NaN",) * 5, regions="ntsfun"),
    "SHRATE": Spec("record", template=("NaN",)),
    # PLYSHLOG is two things at once: a reference-condition record per PVT
    # region, then a table per region. It is the only PROPS keyword whose
    # value is a composite rather than a table or a record.
    "PLYSHLOG": Spec("composite", 2, template=("NaN",) * 3),
    "PLMIXPAR": Spec("record", template=("NaN",) * 2, regions="ntmisc"),
    "TLMIXPAR": Spec("record", template=("NaN",) * 2, regions="ntmisc"),
    # surfactant and miscible
    "SURFST": Spec("immiscible", 2), "SURFVISC": Spec("immiscible", 2),
    "SURFCAPD": Spec("rel_perm", 2, "ntsfun"),
    "SURFADS": Spec("rel_perm", 2, "ntsfun"),
    "SURFROCK": Spec("record", template=("NaN",) * 2, regions="ntsfun"),
    "MISC": Spec("rel_perm", 2, "ntsfun"),
    "PMISC": Spec("rel_perm", 2, "ntsfun"),
    # compositional component vectors
    "MW": Spec("vector"), "PCRIT": Spec("vector"), "TCRIT": Spec("vector"),
    "VCRIT": Spec("vector"), "ZCRIT": Spec("vector"),
    "PARACHOR": Spec("vector"), "SSHIFT": Spec("vector"), "ACF": Spec("vector"),
}


@dataclass
class TableCounts:
    """How many tables each family has, from RUNSPEC's TABDIMS."""

    ntpvt: int = 1
    ntsfun: int = 1
    ntrocc: int = 1
    ntmisc: int = 1
    ncomp: int = 0
    one: int = 1


def _read_vector(cursor: DeckCursor) -> np.ndarray:
    return np.array([float(strip_quotes(t))
                     for t in cursor.read_record_tokens()], dtype=float)


def read_props(cursor: DeckCursor, counts: Optional[TableCounts] = None,
               props: Optional[Dict] = None,
               unhandled: Optional[List[str]] = None) -> Dict[str, Any]:
    """Read the PROPS section into ``{keyword: data}``.

    Keywords outside `SPECS` are appended to ``unhandled`` rather than being
    dropped without trace. MRST's readPROPS covers roughly 320 keywords; most
    of the remainder are the endpoint-scaling family (SWL, KRW, SGCR and their
    directional variants), which are per-cell arrays rather than tables and are
    handled with the relative-permeability scaling work, not here.
    """
    counts = counts or TableCounts()
    out: Dict[str, Any] = dict(props or {})
    unhandled = unhandled if unhandled is not None else []

    while True:
        keyword = cursor.next_keyword()
        if keyword is None:
            break
        if keyword in SECTION_KEYWORDS:
            cursor.rewind_keyword()     # the caller reads the next section
            break
        if keyword in IGNORED or keyword == "ENDBOX":
            continue
        if keyword == "BOX":
            cursor.read_record_tokens()
            continue

        spec = SPECS.get(keyword)
        if spec is None:
            # next_keyword() skips this keyword's data lines, because data
            # never matches the keyword pattern. Record it so a caller can
            # tell "this deck used a keyword we do not read" from "this deck
            # did not use it".
            unhandled.append(keyword)
            continue

        n_tab = getattr(counts, spec.regions, 1)
        if spec.form == "rel_perm":
            out[keyword] = read_rel_perm_table(cursor, keyword, n_tab,
                                               spec.columns)
        elif spec.form == "immiscible":
            out[keyword] = read_immiscible_pvt_table(cursor, n_tab,
                                                     spec.columns)
        elif spec.form == "miscible":
            out[keyword] = read_miscible_pvt_table(cursor, n_tab, spec.columns)
        elif spec.form == "record":
            template = list(spec.template)
            records = (read_fixed_num_records(cursor, template, n_tab)
                       if spec.fixed
                       else read_defaulted_kw(cursor, template, n_rec=n_tab))
            try:
                values = np.array(
                    [[float(v) for v in record] for record in records],
                    dtype=float).reshape(-1, len(template))
            except ValueError as error:
                # Almost always TABDIMS declaring more regions than the deck
                # states tables: the reader runs past this keyword's records
                # into the next keyword's text. Say so, because the raw
                # "could not convert string to float: 'PVTW'" does not.
                raise ValueError(
                    f"{keyword}: could not read {n_tab} record(s). TABDIMS "
                    f"declares {n_tab} {spec.regions} region(s); check the "
                    f"deck states that many {keyword} records ({error})"
                ) from error
            if keyword == "SHRATE":
                # readPROPS substitutes 4.8 for a missing shear rate, and uses
                # it for every region when the keyword carries no records.
                values = (np.full((n_tab, 1), 4.8) if values.size == 0
                          else np.where(np.isnan(values), 4.8, values))
            out[keyword] = values
        elif spec.form == "vector":
            out[keyword] = _read_vector(cursor)
        elif spec.form == "mixed_record":
            record = read_defaulted_record(cursor, list(spec.template))
            for position in spec.numeric:
                record[position] = float(record[position])
            out[keyword] = record
        elif spec.form == "composite":
            reference = read_defaulted_kw(cursor, list(spec.template),
                                          n_rec=n_tab)
            out[keyword] = {
                "refcondition": np.array(
                    [[float(v) for v in record] for record in reference],
                    dtype=float).reshape(-1, len(spec.template)),
                "data": read_immiscible_pvt_table(cursor, n_tab, spec.columns),
            }
    return out


readPROPS = read_props


if __name__ == "__main__":
    from pymrst.modules.deckformat.deckinput.read_defaulted_record import tokenize

    def test_reads_a_saturation_function_table() -> None:
        text = "SWOF\n0.1 0.0 1.0 0.0\n0.9 1.0 0.0 0.0 /\n"
        out = read_props(DeckCursor(tokenize(text)))
        assert out["SWOF"][0].shape == (2, 4), out["SWOF"][0].shape
        print("ok test_reads_a_saturation_function_table")

    def test_reads_a_miscible_pvt_table_packed() -> None:
        text = "PVTO\n0.0 100 1.1 1.0\n 200 1.08 1.1 /\n50 200 1.3 0.9 /\n/\n"
        out = read_props(DeckCursor(tokenize(text)))
        table = out["PVTO"][0]
        assert np.allclose(table.key, [0.0, 50.0]), table.key
        print("ok test_reads_a_miscible_pvt_table_packed")

    def test_reads_a_record_keyword() -> None:
        out = read_props(DeckCursor(tokenize("DENSITY\n900 1000 0.8 /\n")))
        assert np.allclose(out["DENSITY"][0], [900, 1000, 0.8]), out["DENSITY"]
        print("ok test_reads_a_record_keyword")

    def test_record_template_supplies_missing_fields() -> None:
        out = read_props(DeckCursor(tokenize("PVTW\n300 1.0 /\n")))
        assert out["PVTW"].shape == (1, 5), out["PVTW"].shape
        assert np.allclose(out["PVTW"][0, 2:], 0.0), out["PVTW"]
        print("ok test_record_template_supplies_missing_fields")

    def test_multiple_regions_read_multiple_tables() -> None:
        text = "SWOF\n0.1 0.0 1.0 0.0 /\n0.2 0.1 0.9 0.0 /\n"
        out = read_props(DeckCursor(tokenize(text)), TableCounts(ntsfun=2))
        assert len(out["SWOF"]) == 2, len(out["SWOF"])
        assert np.isclose(out["SWOF"][1][0, 0], 0.2)
        print("ok test_multiple_regions_read_multiple_tables")

    def test_several_keywords_in_one_section() -> None:
        text = ("DENSITY\n900 1000 0.8 /\n"
                "SWOF\n0.1 0.0 1.0 0.0 /\n"
                "ROCK\n300 1e-6 /\n")
        out = read_props(DeckCursor(tokenize(text)))
        assert set(out) == {"DENSITY", "SWOF", "ROCK"}, set(out)
        print("ok test_several_keywords_in_one_section")

    def test_stops_at_the_next_section() -> None:
        text = "DENSITY\n900 1000 0.8 /\nREGIONS\nSATNUM\n1 1 /\n"
        cursor = DeckCursor(tokenize(text))
        out = read_props(cursor)
        assert set(out) == {"DENSITY"}
        assert cursor.peek() == "REGIONS"
        print("ok test_stops_at_the_next_section")

    def test_unknown_keyword_does_not_derail_the_cursor() -> None:
        text = "MYSTERY\n1 2 3 /\nDENSITY\n900 1000 0.8 /\n"
        out = read_props(DeckCursor(tokenize(text)))
        assert set(out) == {"DENSITY"}, set(out)
        print("ok test_unknown_keyword_does_not_derail_the_cursor")

    def test_ehystr_keeps_its_mode_words_as_text() -> None:
        """A record mixing numbers and modes cannot become a float array."""
        out = read_props(DeckCursor.from_text("EHYSTR\n0.1 2 1.0 0.1 KR /\n"))
        record = out["EHYSTR"]
        assert record[0] == 0.1 and record[1] == 2.0, record
        assert record[4] == "KR", record
        assert record[5] == "RETR", "the template fills unstated fields"
        print("ok test_ehystr_keeps_its_mode_words_as_text")

    def test_ehystr_feeds_assign_ehystr() -> None:
        from pymrst.modules.ad_props.props.assign_ehystr import assign_ehystr
        out = read_props(DeckCursor.from_text("EHYSTR\n4* BOTH /\n"))
        f = assign_ehystr({}, out["EHYSTR"])
        assert f["krHyst"] == 1 and f["pcHyst"] == 1
        print("ok test_ehystr_feeds_assign_ehystr")

    def test_plyshlog_reads_a_record_and_a_table() -> None:
        # The reference record is followed directly by the table: no
        # terminator between them, because NRec bounds the record read.
        text = ("PLYSHLOG\n0.5 0.0 1.0 /\n"
                "-4.0 1.0\n-2.0 0.6\n0.0 0.3 /\n")
        out = read_props(DeckCursor.from_text(text))["PLYSHLOG"]
        assert np.allclose(out["refcondition"][0], [0.5, 0.0, 1.0])
        assert out["data"][0].shape == (3, 2), out["data"][0].shape
        print("ok test_plyshlog_reads_a_record_and_a_table")

    def test_plyshlog_feeds_assign_plyshlog() -> None:
        from pymrst.modules.ad_props.props.assign_plyshlog import (
            assign_plyshlog,
        )
        text = "PLYSHLOG\n0.5 0.0 1.0 /\n-4.0 1.0\n0.0 0.3 /\n"
        out = read_props(DeckCursor.from_text(text))
        f = assign_plyshlog({}, out["PLYSHLOG"])
        assert "refcondition" in f["plyshlog"] and "data" in f["plyshlog"]
        print("ok test_plyshlog_feeds_assign_plyshlog")

    def test_unhandled_keywords_are_reported() -> None:
        """A keyword we skip must be visible, not silently absent."""
        unhandled = []
        read_props(DeckCursor.from_text("SWL\n0.1 0.1 /\nDENSITY\n900 1 1 /\n"),
                   unhandled=unhandled)
        assert unhandled == ["SWL"], unhandled
        print("ok test_unhandled_keywords_are_reported")

    def test_specs_only_names_keywords_mrst_reads() -> None:
        """Guard against inventing a keyword readPROPS does not have."""
        import pathlib
        import re
        source = pathlib.Path(
            "/Users/Shared/PyMRST_workspace/mrst-2024a-Geomech"
            "/modules/deckformat/deckinput/readPROPS.m")
        if not source.exists():
            print("skip test_specs_only_names_keywords_mrst_reads (no MRST)")
            return
        body = source.read_text()
        body = body[body.index("switch kw"):]
        mrst = set()
        for match in re.finditer(r"^\s*case\s+(\{[^}]*\}|'[^']*')", body, re.M):
            mrst |= set(re.findall(r"'([^']+)'", match.group(1)))
        invented = sorted(set(SPECS) - mrst)
        assert not invented, f"not in MRST's readPROPS: {invented}"
        print(f"ok test_specs_only_names_keywords_mrst_reads "
              f"({len(SPECS)} of {len(mrst)} MRST cases)")

    def test_every_spec_names_a_known_form() -> None:
        forms = {"rel_perm", "immiscible", "miscible", "record", "vector",
                 "mixed_record", "composite"}
        for keyword, spec in SPECS.items():
            assert spec.form in forms, (keyword, spec.form)
            if spec.form in ("rel_perm", "immiscible", "miscible"):
                assert spec.columns, keyword
            if spec.form in ("record", "mixed_record", "composite"):
                assert spec.template, keyword
        print(f"ok test_every_spec_names_a_known_form ({len(SPECS)} keywords)")

    test_reads_a_saturation_function_table()
    test_reads_a_miscible_pvt_table_packed()
    test_reads_a_record_keyword()
    test_record_template_supplies_missing_fields()
    test_multiple_regions_read_multiple_tables()
    test_several_keywords_in_one_section()
    test_stops_at_the_next_section()
    test_unknown_keyword_does_not_derail_the_cursor()
    test_ehystr_keeps_its_mode_words_as_text()
    test_ehystr_feeds_assign_ehystr()
    test_plyshlog_reads_a_record_and_a_table()
    test_plyshlog_feeds_assign_plyshlog()
    test_unhandled_keywords_are_reported()
    test_every_spec_names_a_known_form()
    test_specs_only_names_keywords_mrst_reads()
    print("\nAll tests passed!")
