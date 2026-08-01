"""Record operator entries without applying them (MRST ``.../private/readOperator.m``).

Reads an `ADD`/`MULTIPLY`/`EQUALS`/... section and stores each entry under
`grdecl[keyword][target]` as parallel `value` and `region` lists, rather than
modifying the array. `read_operator_mod_field` is the variant that applies them
immediately, and `apply_operator_simple` is what does the applying.

**Why keep the deferred form.** The order of operations matters - `MULTIPLY`
then `ADD` is not `ADD` then `MULTIPLY` - and a caller reading a deck in
sections may not have the target array yet when the operator appears. Recording
first and applying later is how a reader stays independent of keyword order.

MRST's arity check here is `if(~length(split)==8)`, which parses as
`(~length(split)) == 8` and is never true - the same dead check as in
`readFaults`, `readMultfelt` and `readOperatorModField`, four instances of one
mistake. Records are checked properly here.
"""

import re
from typing import Dict, Iterable, Optional

import numpy as np

from pymrst.utils.inout.eclipse.deckinput_simple.apply_operator_simple import (
    OPERATORS,
)

TOKEN = re.compile(r"[\w.\-*+]+")


def read_operator(lines: Iterable[str], grdecl: Optional[Dict] = None,
                  kw: str = "MULTIPLY") -> Dict:
    """Collect one operator section into ``grdecl[kw]``."""
    kw = kw.upper()
    if kw not in OPERATORS:
        raise ValueError(f"{kw!r} is not a box operator; expected {OPERATORS}")

    deck = {} if grdecl is None else grdecl
    section = deck.setdefault(kw, {})

    for raw in lines:
        line = raw.strip()
        if line.startswith("/"):
            break
        if not line or line.startswith("--"):
            continue

        fields = TOKEN.findall(line)
        if len(fields) != 8:
            raise ValueError(
                f"a {kw} line needs a target, a value and six box indices "
                f"(8 fields), got {len(fields)}: {line!r}")
        name = fields[0]
        try:
            value = float(fields[1])
            region = [int(v) for v in fields[2:8]]
        except ValueError:
            raise ValueError(f"a {kw} line has a non-numeric field: {line!r}") \
                from None

        entry = section.setdefault(name, {"value": [], "region": []})
        entry["value"].append(value)
        entry["region"].append(region)

    for entry in section.values():
        entry["value"] = np.asarray(entry["value"], dtype=float)
        entry["region"] = np.asarray(entry["region"], dtype=np.int64)
    return deck


readOperator = read_operator


if __name__ == "__main__":

    def test_it_records_one_entry() -> None:
        deck = read_operator(["'PERMX' 0.1 1 2 1 2 1 1 /"], kw="MULTIPLY")
        entry = deck["MULTIPLY"]["PERMX"]
        assert entry["value"].tolist() == [0.1]
        assert entry["region"].tolist() == [[1, 2, 1, 2, 1, 1]]
        print("ok test_it_records_one_entry")

    def test_entries_accumulate_per_target() -> None:
        deck = read_operator(["'PERMX' 0.1 1 2 1 2 1 1",
                              "'PERMX' 3.0 1 2 1 2 2 2", "/"], kw="MULTIPLY")
        entry = deck["MULTIPLY"]["PERMX"]
        assert entry["value"].tolist() == [0.1, 3.0]
        assert entry["region"].shape == (2, 6)
        print("ok test_entries_accumulate_per_target")

    def test_targets_stay_separate() -> None:
        deck = read_operator(["'PERMX' 0.1 1 2 1 2 1 1",
                              "'PORO' 0.5 1 2 1 2 1 1", "/"], kw="MULTIPLY")
        assert sorted(deck["MULTIPLY"]) == ["PERMX", "PORO"]
        print("ok test_targets_stay_separate")

    def test_nothing_is_applied() -> None:
        """The point of the deferred form: the array is untouched."""
        deck = {"PERMX": np.full(4, 100.0)}
        read_operator(["'PERMX' 0.1 1 2 1 2 1 1 /"], deck, kw="MULTIPLY")
        assert deck["PERMX"].tolist() == [100.0] * 4
        print("ok test_nothing_is_applied")

    def test_it_stops_at_the_terminator() -> None:
        lines = iter(["'PERMX' 0.1 1 2 1 2 1 1", "/", "PORO"])
        read_operator(lines, kw="MULTIPLY")
        assert next(lines) == "PORO"
        print("ok test_it_stops_at_the_terminator")

    def test_blanks_and_comments_are_skipped() -> None:
        deck = read_operator(["", "-- note", "'PERMX' 0.1 1 2 1 2 1 1", "/"],
                             kw="ADD")
        assert deck["ADD"]["PERMX"]["value"].tolist() == [0.1]
        print("ok test_blanks_and_comments_are_skipped")

    def test_the_records_can_then_be_applied() -> None:
        """Deferred and immediate must agree."""
        from pymrst.utils.inout.eclipse.deckinput_simple.apply_operator_simple import (
            apply_operator_simple,
        )
        deck = read_operator(["'PERMX' 0.1 1 2 1 2 1 1 /"], kw="MULTIPLY")
        sect = {"PERMX": np.full(4, 100.0)}
        entry = deck["MULTIPLY"]["PERMX"]
        records = [["PERMX", v, *r] for v, r in
                   zip(entry["value"], entry["region"].tolist())]
        got = apply_operator_simple(sect, records, "MULTIPLY", (2, 2, 1))
        assert got["PERMX"].tolist() == [10.0] * 4
        print("ok test_the_records_can_then_be_applied")

    def test_a_short_line_is_rejected() -> None:
        """MRST's check never fires; see the module docstring."""
        try:
            read_operator(["'PERMX' 0.1 1 2 /"], kw="MULTIPLY")
        except ValueError as error:
            assert "8 fields" in str(error)
            print("ok test_a_short_line_is_rejected")
            return
        raise AssertionError("a malformed operator line must be rejected")

    def test_an_unknown_operator_is_rejected() -> None:
        try:
            read_operator([], kw="SCALE")
        except ValueError as error:
            assert "not a box operator" in str(error)
            print("ok test_an_unknown_operator_is_rejected")
            return
        raise AssertionError("an unknown operator must be rejected")

    test_it_records_one_entry()
    test_entries_accumulate_per_target()
    test_targets_stay_separate()
    test_nothing_is_applied()
    test_it_stops_at_the_terminator()
    test_blanks_and_comments_are_skipped()
    test_the_records_can_then_be_applied()
    test_a_short_line_is_rejected()
    test_an_unknown_operator_is_rejected()
    print("\nAll tests passed!")
