"""Read a FAULTS record (MRST ``.../deckinput_simple/private/readFaults.m``).

Each line names a fault and the index box it occupies:

    'F1'  10 10  1 20  1 5  X /

giving `(i1, i2, j1, j2, k1, k2)` and a face direction. A fault usually spans
several boxes, so its name repeats and the boxes accumulate.

**Indices are converted to 0-based**, as everywhere in PyMRST, and the box is
inclusive at both ends - ECLIPSE's convention, kept.

**MRST's arity check never fires.** It reads

    if(~length(split)==8)

which MATLAB parses as `(~length(split)) == 8`; for any non-empty split that is
`0 == 8`, so a malformed line falls through and is silently mis-parsed - or
raises something unrelated further down. (Its error message says
"readOpertor", which is a second clue nobody has ever seen printed.) This
checks the count properly and names the offending line.
"""

import re
from typing import Dict, Iterable, Optional

import numpy as np

TOKEN = re.compile(r"[\w.\-*]+")


def read_faults(lines: Iterable[str], grdecl: Optional[Dict] = None) -> Dict:
    """Consume a FAULTS section from ``lines``; return the updated deck."""
    deck = {} if grdecl is None else grdecl
    faults = deck.setdefault("FAULTS", {})

    for raw in lines:
        line = raw.strip()
        if line.startswith("/"):
            break
        if not line or line.startswith("--"):
            continue

        fields = TOKEN.findall(line)
        if len(fields) != 8:
            raise ValueError(
                f"a FAULTS line needs a name, six indices and a direction "
                f"(8 fields), got {len(fields)}: {line!r}")

        name, direction = fields[0], fields[7]
        try:
            box = np.array([int(v) for v in fields[1:7]], dtype=np.int64) - 1
        except ValueError:
            raise ValueError(
                f"a FAULTS line has a non-integer index: {line!r}") from None
        if np.any(box < 0):
            raise ValueError(
                f"a FAULTS index is below 1; ECLIPSE numbers from 1: {line!r}")

        entry = faults.setdefault(name, {"cells": [], "dir": []})
        entry["cells"].append(box)
        entry["dir"].append(direction)

    for entry in faults.values():
        if isinstance(entry["cells"], list) and entry["cells"]:
            entry["cells"] = np.vstack(entry["cells"])
    return deck


readFaults = read_faults


if __name__ == "__main__":

    def test_one_fault_one_box() -> None:
        deck = read_faults(["'F1' 10 10 1 20 1 5 X /"])
        assert list(deck["FAULTS"]) == ["F1"]
        # 0-based, so ECLIPSE's 10 10 1 20 1 5 becomes 9 9 0 19 0 4.
        assert deck["FAULTS"]["F1"]["cells"].tolist() == [[9, 9, 0, 19, 0, 4]]
        assert deck["FAULTS"]["F1"]["dir"] == ["X"]
        print("ok test_one_fault_one_box")

    def test_a_fault_accumulates_its_boxes() -> None:
        deck = read_faults([
            "'F1' 10 10 1 20 1 5 X",
            "'F1' 11 11 1 20 6 9 Y",
            "/",
        ])
        assert deck["FAULTS"]["F1"]["cells"].shape == (2, 6)
        assert deck["FAULTS"]["F1"]["dir"] == ["X", "Y"]
        print("ok test_a_fault_accumulates_its_boxes")

    def test_several_faults_stay_separate() -> None:
        deck = read_faults(["'F1' 1 1 1 2 1 3 X", "'F2' 4 4 1 2 1 3 Y", "/"])
        assert sorted(deck["FAULTS"]) == ["F1", "F2"]
        assert deck["FAULTS"]["F2"]["cells"].tolist() == [[3, 3, 0, 1, 0, 2]]
        print("ok test_several_faults_stay_separate")

    def test_it_stops_at_the_terminator() -> None:
        lines = iter(["'F1' 1 1 1 2 1 3 X", "/", "PORO", "0.2"])
        read_faults(lines)
        assert next(lines) == "PORO", "the reader consumed past the '/'"
        print("ok test_it_stops_at_the_terminator")

    def test_blanks_and_comments_are_skipped() -> None:
        deck = read_faults(["", "-- a note", "  ", "'F1' 1 1 1 2 1 3 X", "/"])
        assert deck["FAULTS"]["F1"]["cells"].shape == (1, 6)
        print("ok test_blanks_and_comments_are_skipped")

    def test_it_adds_to_an_existing_deck() -> None:
        deck = read_faults(["'F1' 1 1 1 2 1 3 X", "/"], {"PORO": [0.2]})
        assert "PORO" in deck and "FAULTS" in deck
        print("ok test_it_adds_to_an_existing_deck")

    def test_a_short_line_is_rejected() -> None:
        """MRST's check is `(~length(split)) == 8` and never fires."""
        try:
            read_faults(["'F1' 10 10 1 20 X /"])
        except ValueError as error:
            assert "8 fields" in str(error)
            print("ok test_a_short_line_is_rejected")
            return
        raise AssertionError("a malformed FAULTS line must be rejected")

    def test_a_non_integer_index_is_rejected() -> None:
        try:
            read_faults(["'F1' 10 ten 1 20 1 5 X /"])
        except ValueError as error:
            assert "non-integer index" in str(error)
            print("ok test_a_non_integer_index_is_rejected")
            return
        raise AssertionError("a non-integer index must be rejected")

    def test_a_zero_index_is_rejected() -> None:
        """ECLIPSE numbers from 1, so 0 means the deck is wrong."""
        try:
            read_faults(["'F1' 0 10 1 20 1 5 X /"])
        except ValueError as error:
            assert "numbers from 1" in str(error)
            print("ok test_a_zero_index_is_rejected")
            return
        raise AssertionError("a zero index must be rejected")

    test_one_fault_one_box()
    test_a_fault_accumulates_its_boxes()
    test_several_faults_stay_separate()
    test_it_stops_at_the_terminator()
    test_blanks_and_comments_are_skipped()
    test_it_adds_to_an_existing_deck()
    test_a_short_line_is_rejected()
    test_a_non_integer_index_is_rejected()
    test_a_zero_index_is_rejected()
    print("\nAll tests passed!")
