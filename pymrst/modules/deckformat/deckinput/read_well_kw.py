"""Well keywords in a deck's SCHEDULE section (MRST ``deckinput/readWellKW.m``).

Every keyword here is a table of `/`-terminated records with a *template*: the
list of defaults a field takes when the deck writes `1*` or leaves it off the
end. The templates below are transcribed from `readWellKW.m` rather than
inferred, because a wrong default is invisible - `WCONPROD` column 9 defaulting
to `0` instead of `NaN` gives every producer a zero bottom-hole limit and the
run simply produces nothing.

Two behaviours are less obvious than the templates and matter as much:

* **a control keyword replaces the record it matches.** `WCONINJE` for a well
  that already has a `WCONPROD` record must *remove* the producer record, not
  sit alongside it. MRST calls this the exclude set; without it a well carries
  two contradictory controls and which one wins depends on iteration order;
* **`COMPDAT` merges rather than appends.** A later record covering the same
  well and the same layer range modifies that completion; one covering a new
  range adds to it. Appending unconditionally perforates the same cell twice
  and doubles its well index.

Coverage is the keyword set real decks use. Anything else is reported through
`unsupported`, because a silently dropped well keyword changes the answer.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from pymrst.utils.inout.eclipse.deckinput_simple.read_defaulted_record import (
    DeckCursor, read_defaulted_kw,
)

# `readWellKW.m`'s templates, with the 1-based `numeric` column lists it uses
# turned into 0-based indices.
TEMPLATES: Dict[str, Sequence[str]] = {
    "WELSPECS": ("Default", "Default", "Default", "Default", "NaN",
                 "Default", "0.0", "STD", "SHUT", "YES", "0", "SEG", "0",
                 "FrontSim", "FrontSim", "STD"),
    "COMPDAT": ("Default", "-1", "-1", "Default", "Default", "OPEN", "-1",
                "-1.0", "0.0", "-1.0", "0.0", "Default", "Z", "-1"),
    "WCONPROD": ("Default", "OPEN", "Default", "inf", "inf", "inf", "inf",
                 "inf", "NaN", "0.0", "0", "0.0"),
    "WCONHIST": ("Default", "OPEN", "Default", "0.0", "0.0", "0.0", "0",
                 "Default", "0.0", "0.0", "0.0"),
    "WCONINJE": ("Default", "Default", "OPEN", "Default", "inf", "inf",
                 "NaN", "inf", "0", "0.0", "0.0", "0", "0", "0"),
    "WCONINJH": ("Default", "Default", "OPEN", "0.0", "0.0", "0.0", "NaN",
                 "0", "0", "0", "0"),
    "WCONINJ": ("Default", "Default", "OPEN", "Default", "inf", "inf",
                "NaN", "inf", "0", "0.0"),
    "WELOPEN": ("Default", "OPEN", "-1", "-1", "-1", "-1", "-1"),
    "WELTARG": ("Default", "Default", "NaN"),
    "WPIMULT": ("Default", "1.0", "-1", "-1", "-1", "-1", "-1"),
    "WPOLYMER": ("Default", "0.0", "0.0", "Default", "Default"),
    "WSOLVENT": ("Default", "0.0"),
    "WSURFACT": ("Default", "0.0"),
    "WTEMP": ("Default", "NaN"),
    "GRUPTREE": ("Default", "Default"),
}

NUMERIC: Dict[str, Sequence[int]] = {
    "WELSPECS": (2, 3, 4, 6, 10, 12),
    "COMPDAT": (1, 2, 3, 4, 6, 7, 8, 9, 10, 13),
    "WCONPROD": tuple(range(3, 12)),
    "WCONHIST": (3, 4, 5, 6, 8, 9, 10),
    "WCONINJE": tuple(range(4, 14)),
    "WCONINJH": tuple(range(3, 11)),
    "WCONINJ": tuple(range(4, 10)),
    "WELOPEN": tuple(range(2, 7)),
    "WELTARG": (2,),
    "WPIMULT": tuple(range(1, 7)),
    "WPOLYMER": (1, 2),
    "WSOLVENT": (1,),
    "WSURFACT": (1,),
    "WTEMP": (1,),
    "GRUPTREE": (),
}

#: The mutually exclusive operating-control keywords. Setting one for a well
#: removes that well's record from all the others (MRST `excludeSet`).
CONTROL_KEYWORDS = ("WCONINJ", "WCONINJE", "WCONINJH", "WCONPROD", "WCONHIST")

#: Per-well keywords MRST merges with `appendSpec` (wildcards expanded).
APPEND_SPEC_KEYWORDS = ("WPOLYMER", "WSURFACT", "WSOLVENT", "WTEMP")

#: Read as a report-control record and otherwise ignored, as in MRST.
REPORT_KEYWORDS = ("RPTSCHED", "RPTRST", "OUTSOL")

#: Keywords that may appear before any WELSPECS has declared a well.
NO_WELL_OK = ("GRUPTREE", "WELSPECS") + REPORT_KEYWORDS

SUPPORTED = tuple(TEMPLATES) + REPORT_KEYWORDS


def default_control() -> Dict[str, Any]:
    """An empty control period (MRST ``defaultControl``)."""
    control: Dict[str, Any] = {name: [] for name in TEMPLATES}
    control.update({name: [] for name in REPORT_KEYWORDS})
    control["VAPPARS"] = [0.0, 0.0]
    control["DRSDT"] = [float("inf"), "ALL"]
    return control


#: Fields a new control period inherits verbatim: the well geometry, which
#: persists until the deck restates it.
_CARRIED = ("WELSPECS", "COMPDAT", "GRUPTREE")
#: Fields a new period inherits only if the previous one set them - the
#: operating controls, which also persist but are replaced wholesale.
_CARRIED_IF_SET = CONTROL_KEYWORDS + REPORT_KEYWORDS


def next_control(previous: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The next control period, carrying forward what ECLIPSE carries forward.

    A deck states a well's controls once and changes only what moves. Starting
    each period empty would shut in every well the deck stopped mentioning.
    """
    control = default_control()
    if previous is None:
        return control
    for name in _CARRIED:
        control[name] = [list(record) for record in previous.get(name, [])]
    control["DRSDT"] = list(previous.get("DRSDT", control["DRSDT"]))
    control["VAPPARS"] = list(previous.get("VAPPARS", control["VAPPARS"]))
    for name in _CARRIED_IF_SET:
        if previous.get(name):
            control[name] = [list(record) if isinstance(record, list) else record
                             for record in previous[name]]
    return control


def read_well_kw(cursor: DeckCursor, control: Dict[str, Any],
                 keyword: str) -> Dict[str, Any]:
    """Read one well keyword into ``control`` and return it."""
    if keyword in REPORT_KEYWORDS:
        control[keyword] = cursor.read_record_tokens()
        return control
    if keyword not in TEMPLATES:
        raise NotImplementedError(
            f"well keyword {keyword!r} is not supported by this reader; it "
            "would change the schedule and is not silently dropped")
    if keyword not in NO_WELL_OK and not control["WELSPECS"]:
        raise ValueError(
            f"well keyword {keyword!r} appears before any well has been "
            "declared with WELSPECS")

    records = _to_double(read_defaulted_kw(cursor, TEMPLATES[keyword]),
                         NUMERIC[keyword])
    if not records:
        return control

    if keyword == "WELSPECS":
        control["WELSPECS"] = _replace_by_name(control["WELSPECS"], records)
    elif keyword == "COMPDAT":
        control["COMPDAT"] = _merge_compdat(
            control["COMPDAT"], _expand_patterns(records, control["WELSPECS"]))
    elif keyword in CONTROL_KEYWORDS:
        # MRST's assignControlRecords: appendSpec, then drop the same wells'
        # records from the other control keywords.
        wells = [str(spec[0]) for spec in control["WELSPECS"]]
        control[keyword] = _append_spec(control.get(keyword) or [], keyword, records, wells)
        names = {str(record[0]) for record in records}
        for other in CONTROL_KEYWORDS:
            if other != keyword and control.get(other):
                control[other] = [r for r in control[other] if str(r[0]) not in names]
    elif keyword in APPEND_SPEC_KEYWORDS:
        wells = [str(spec[0]) for spec in control["WELSPECS"]]
        control[keyword] = _append_spec(control.get(keyword) or [], keyword, records, wells)
    else:
        control[keyword] = list(control.get(keyword) or []) + records
    return control


readWellKW = read_well_kw
defaultControl = default_control


def _name(record: Sequence[Any]) -> str:
    return str(record[0]).strip().upper()


def _to_double(records: List[List[Any]],
               columns: Sequence[int]) -> List[List[Any]]:
    """Convert the template's numeric columns, as MRST's ``toDouble`` does."""
    out = []
    for record in records:
        row = list(record)
        for column in columns:
            if column < len(row):
                row[column] = _float(row[column])
        out.append(row)
    return out


def _float(token: Any) -> float:
    if isinstance(token, (int, float)):
        return float(token)
    text = str(token).strip()
    if text.lower() in ("default", "", "1*"):
        return float("nan")
    # FORTRAN exponent markers: 1.0D+3 is a double literal, not a name.
    try:
        return float(re.sub(r"[dD]", "e", text))
    except ValueError:
        return float("nan")


def _replace_by_name(existing: List[List[Any]],
                     records: List[List[Any]]) -> List[List[Any]]:
    """New records replace same-named old ones in place, and append otherwise.

    Replacing in place rather than appending keeps well order stable across
    control periods, which is what lets a caller compare well `k` in one
    period with well `k` in the next.
    """
    out = [list(record) for record in existing]
    index = {_name(record): position for position, record in enumerate(out)}
    for record in records:
        name = _name(record)
        if name in index:
            out[index[name]] = list(record)
        else:
            index[name] = len(out)
            out.append(list(record))
    return out


def _append_spec(table: List[List[Any]], kw: str, data: List[List[Any]],
                 wells: Sequence[str]) -> List[List[Any]]:
    """MRST `appendSpec`: one record per well. A wildcard record (`'PROD*'`)
    replaces the earlier records it matches and is copied to every declared
    well it matches. Records for named wells replace those wells' earlier
    records and go to the end."""
    import warnings
    matches = lambda name, patt: re.search(patt, str(name)) is not None
    table = [list(r) for r in table]
    data = _unique_well_records(data, kw)
    is_wc = [matches(r[0], r"\w+\*\s*$") for r in data]
    for record, wc in zip(data, is_wc):
        if not wc:
            continue
        patt = str(record[0]).replace("*", ".*")
        table = [r for r in table if not matches(r[0], patt)]
        match_wells = [w for w in wells if matches(w, patt)]
        if not match_wells:
            warnings.warn(f"Well control wildcard '{record[0]}' does not match any "
                          "previously defined wells.  Ignored.")
        for well in match_wells:
            row = list(record)
            row[0] = well
            table.append(row)
    data = [r for r, wc in zip(data, is_wc) if not wc]
    if data and not all(str(r[0]) in wells for r in data):
        unknown = " ".join(f"'{r[0]}'" for r in data if str(r[0]) not in wells)
        warnings.warn(f"Well control specified in undeclared wells: {unknown}.\n"
                      "Check input to verify that well was declared using WELSPECS.")
        return table
    names = {str(r[0]) for r in data}
    return [r for r in table if str(r[0]) not in names] + [list(r) for r in data]


def _unique_well_records(data: List[List[Any]], kw: str) -> List[List[Any]]:
    """MRST `unique_well_records`: the last record for each well, in order."""
    import warnings
    last = {str(r[0]): k for k, r in enumerate(data)}
    if len(last) < len(data):
        repeated = sorted({str(r[0]) for k, r in enumerate(data) if last[str(r[0])] != k})
        warnings.warn(f"{kw}: Multiple Records Apply to Same Well/Well-List.\n"
                      f"  -> Using Last Entry for {{ {' '.join(repeated)} }}.")
    return [r for k, r in enumerate(data) if last[str(r[0])] == k]


def _expand_patterns(records: List[List[Any]],
                     welspecs: List[List[Any]]) -> List[List[Any]]:
    """Expand a `'INJ*'` well name into every declared well it matches."""
    names = [_name(spec) for spec in welspecs]
    out: List[List[Any]] = []
    for record in records:
        raw = str(record[0]).strip()
        if "*" not in raw:
            out.append(list(record))
            continue
        pattern = re.compile("^" + raw.upper().replace("*", ".*") + "$")
        matched = [name for name in names if pattern.match(name)]
        if not matched:
            raise ValueError(
                f"COMPDAT well pattern {raw!r} matches no well in WELSPECS")
        for name in matched:
            expanded = list(record)
            expanded[0] = name
            out.append(expanded)
    return out


def _layers(record: Sequence[Any]) -> tuple:
    """The `(I, J, K1, K2)` box a COMPDAT record perforates."""
    return tuple(_float(record[column]) for column in (1, 2, 3, 4))


def _merge_compdat(existing: List[List[Any]],
                   records: List[List[Any]]) -> List[List[Any]]:
    """Add new completions; replace ones covering the same box.

    MRST's `handleOverlapCompdat` decides this per well. The distinction that
    matters is the one it makes first: a record whose `(I, J, K1, K2)` box
    already exists *modifies* that completion, and appending it instead
    perforates the same cell twice - doubling its well index without any
    complaint from the assembly.
    """
    out = [list(record) for record in existing]
    index = {(_name(record), _layers(record)): position
             for position, record in enumerate(out)}
    for record in records:
        key = (_name(record), _layers(record))
        if key in index:
            out[index[key]] = list(record)
        else:
            index[key] = len(out)
            out.append(list(record))
    return out


__all__ = ["CONTROL_KEYWORDS", "NUMERIC", "REPORT_KEYWORDS", "SUPPORTED",
           "TEMPLATES", "default_control", "defaultControl", "next_control",
           "readWellKW", "read_well_kw"]


if __name__ == "__main__":
    from pymrst.utils.inout.eclipse.deckinput_simple.read_defaulted_record import (
        tokenize_with_lines,
    )

    def _cursor(text: str) -> DeckCursor:
        tokens, lines = tokenize_with_lines(text)
        return DeckCursor(tokens, lines=lines)

    def _read(text: str, control=None):
        control = control if control is not None else default_control()
        cursor = _cursor(text)
        keyword = cursor.next_keyword()
        return read_well_kw(cursor, control, keyword)

    SPECS = ("WELSPECS\n 'PROD1' 'G' 16 43 1* 'OIL' /\n"
             " 'INJ1' 'G' 5 57 1* 'WATER' /\n/\n")

    def test_welspecs_positions_are_numbers():
        control = _read(SPECS)
        assert len(control["WELSPECS"]) == 2
        assert control["WELSPECS"][0][2] == 16.0
        assert control["WELSPECS"][0][3] == 43.0
        print("ok test_welspecs_positions_are_numbers")

    def test_a_defaulted_wconprod_bhp_limit_is_nan_not_zero():
        """The template entry whose value cannot be guessed.

        Column 9 defaults to `NaN`, meaning "no limit". Defaulting it to `0`
        instead gives every producer a zero bottom-hole limit, and the well
        produces nothing while the run converges perfectly.
        """
        control = _read("WCONPROD\n 'PROD1' 'OPEN' 'BHP' 5* 395 /\n/\n",
                        _read(SPECS))
        record = control["WCONPROD"][0]
        assert record[8] == 395.0, record
        assert str(TEMPLATES["WCONPROD"][8]) == "NaN"
        print("ok test_a_defaulted_wconprod_bhp_limit_is_nan_not_zero")

    def test_a_repeated_default_run_shifts_later_fields():
        """`5*` consumes five slots. Reading it as one drops the target four
        columns to the left, where it becomes a liquid rate."""
        control = _read("WCONPROD\n 'PROD1' 'OPEN' 'BHP' 5* 395 /\n/\n",
                        _read(SPECS))
        record = control["WCONPROD"][0]
        assert record[3] == float("inf"), record[3]
        print("ok test_a_repeated_default_run_shifts_later_fields")

    def test_an_injector_control_removes_the_producer_record():
        """The exclude set. Left in place, the well carries two contradictory
        controls and which one wins depends on iteration order."""
        control = _read(SPECS)
        control = _read("WCONPROD\n 'PROD1' 'OPEN' 'BHP' 5* 395 /\n/\n", control)
        control = _read("WCONINJE\n 'PROD1' 'WATER' 'OPEN' 'RATE' 79.5 1* 420 /\n/\n",
                        control)
        assert control["WCONPROD"] == [], control["WCONPROD"]
        assert len(control["WCONINJE"]) == 1
        print("ok test_an_injector_control_removes_the_producer_record")

    def test_wconinje_reads_its_rate_and_bhp_columns():
        control = _read("WCONINJE\n 'INJ1' 'WATER' 'OPEN' 'RATE' 79.5 1* 420 /\n/\n",
                        _read(SPECS))
        record = control["WCONINJE"][0]
        assert record[1] == "WATER" and record[3] == "RATE"
        assert record[4] == 79.5, record[4]
        assert record[6] == 420.0, record[6]
        print("ok test_wconinje_reads_its_rate_and_bhp_columns")

    def test_compdat_defaults_i_and_j_to_minus_one():
        """`2*` leaves the position to WELSPECS; -1 is the marker for that."""
        control = _read("COMPDAT\n 'PROD1' 2* 1 7 'OPEN' 2* 0.2 1* 0 /\n/\n",
                        _read(SPECS))
        record = control["COMPDAT"][0]
        assert record[1] == -1.0 and record[2] == -1.0, record
        assert record[3] == 1.0 and record[4] == 7.0
        assert record[8] == 0.2, record[8]
        print("ok test_compdat_defaults_i_and_j_to_minus_one")

    def test_a_repeated_compdat_box_replaces_rather_than_appends():
        """Appending perforates the same cells twice and doubles the well
        index, with nothing downstream able to notice."""
        text = "COMPDAT\n 'PROD1' 2* 1 7 'OPEN' 2* 0.2 1* 0 /\n/\n"
        control = _read(text, _read(SPECS))
        control = _read(text.replace("0.2", "0.3"), control)
        assert len(control["COMPDAT"]) == 1, control["COMPDAT"]
        assert control["COMPDAT"][0][8] == 0.3
        print("ok test_a_repeated_compdat_box_replaces_rather_than_appends")

    def test_a_new_layer_range_adds_a_completion():
        control = _read("COMPDAT\n 'PROD1' 2* 1 3 'OPEN' /\n/\n", _read(SPECS))
        control = _read("COMPDAT\n 'PROD1' 2* 4 7 'OPEN' /\n/\n", control)
        assert len(control["COMPDAT"]) == 2
        print("ok test_a_new_layer_range_adds_a_completion")

    def test_a_wildcard_control_covers_every_matching_well():
        """MRST's appendSpec: 'P*' becomes one WCONPROD record per declared
        producer, in WELSPECS order."""
        specs = ("WELSPECS\n 'P1' 'G' 1 1 1* 'OIL' /\n 'P2' 'G' 2 2 1* 'OIL' /\n"
                 " 'I1' 'G' 3 3 1* 'WATER' /\n/\n")
        control = _read(specs)
        control = _read("WCONPROD\n 'P*' 'OPEN' 'ORAT' 100 /\n/\n", control)
        assert [r[0] for r in control["WCONPROD"]] == ["P1", "P2"]
        assert all(r[3] == 100.0 for r in control["WCONPROD"])
        print("ok test_a_wildcard_control_covers_every_matching_well")

    def test_a_named_record_moves_its_well_to_the_end():
        """appendSpec drops a well's old record and appends the new one."""
        specs = ("WELSPECS\n 'P1' 'G' 1 1 1* 'OIL' /\n 'P2' 'G' 2 2 1* 'OIL' /\n/\n")
        control = _read(specs)
        control = _read("WCONPROD\n 'P*' 'OPEN' 'ORAT' 100 /\n/\n", control)
        control = _read("WCONPROD\n 'P1' 'OPEN' 'ORAT' 50 /\n/\n", control)
        assert [(r[0], r[3]) for r in control["WCONPROD"]] == [("P2", 100.0), ("P1", 50.0)]
        print("ok test_a_named_record_moves_its_well_to_the_end")

    def test_a_wildcard_expands_over_declared_wells():
        control = _read("COMPDAT\n 'INJ*' 2* 1 7 'OPEN' /\n/\n", _read(SPECS))
        assert [_name(r) for r in control["COMPDAT"]] == ["INJ1"]
        print("ok test_a_wildcard_expands_over_declared_wells")

    def test_a_control_before_any_well_is_refused():
        try:
            _read("WCONPROD\n 'PROD1' 'OPEN' 'BHP' 5* 395 /\n/\n")
        except ValueError as refused:
            assert "WELSPECS" in str(refused)
            print("ok test_a_control_before_any_well_is_refused")
            return
        raise AssertionError("a control keyword needs a declared well")

    def test_an_unsupported_well_keyword_is_refused_not_dropped():
        try:
            _read("WFOOBAR\n 'PROD1' 1 /\n/\n", _read(SPECS))
        except NotImplementedError as refused:
            assert "WFOOBAR" in str(refused)
            print("ok test_an_unsupported_well_keyword_is_refused_not_dropped")
            return
        raise AssertionError("an unknown well keyword must not be dropped")

    def test_a_new_period_inherits_geometry_and_controls():
        control = _read("WCONPROD\n 'PROD1' 'OPEN' 'BHP' 5* 395 /\n/\n",
                        _read("COMPDAT\n 'PROD1' 2* 1 7 'OPEN' /\n/\n",
                              _read(SPECS)))
        following = next_control(control)
        assert len(following["WELSPECS"]) == 2
        assert len(following["COMPDAT"]) == 1
        assert following["WCONPROD"][0][8] == 395.0
        print("ok test_a_new_period_inherits_geometry_and_controls")

    def test_a_carried_period_does_not_alias_the_previous_one():
        control = _read("COMPDAT\n 'PROD1' 2* 1 7 'OPEN' /\n/\n", _read(SPECS))
        following = next_control(control)
        following["COMPDAT"][0][8] = 99.0
        assert control["COMPDAT"][0][8] != 99.0
        print("ok test_a_carried_period_does_not_alias_the_previous_one")

    test_welspecs_positions_are_numbers()
    test_a_defaulted_wconprod_bhp_limit_is_nan_not_zero()
    test_a_repeated_default_run_shifts_later_fields()
    test_an_injector_control_removes_the_producer_record()
    test_wconinje_reads_its_rate_and_bhp_columns()
    test_compdat_defaults_i_and_j_to_minus_one()
    test_a_repeated_compdat_box_replaces_rather_than_appends()
    test_a_new_layer_range_adds_a_completion()
    test_a_wildcard_control_covers_every_matching_well()
    test_a_named_record_moves_its_well_to_the_end()
    test_a_wildcard_expands_over_declared_wells()
    test_a_control_before_any_well_is_refused()
    test_an_unsupported_well_keyword_is_refused_not_dropped()
    test_a_new_period_inherits_geometry_and_controls()
    test_a_carried_period_does_not_alias_the_previous_one()
    print("\nAll tests passed!")
