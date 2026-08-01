"""Read a MULTFLT record (MRST ``.../deckinput_simple/private/readMultfelt.m``).

Assigns a transmissibility multiplier to a fault that `read_faults` already
named:

    'F1'  0.001 /

so it must run **after** FAULTS. MRST raises "Fault multiplier for not named
fault" otherwise, and so does this - a multiplier for an unknown fault is
almost always a typo in the name, and silently creating the fault would hide it.

**MRST's arity check never fires**, the same defect as in `readFaults`: it
reads `if(~length(split)==2)`, which MATLAB parses as
`(~length(split)) == 2` - always false for a non-empty line. A line with the
wrong number of fields therefore reaches `str2num` and fails obscurely or,
worse, quietly assigns the wrong value. This checks the count and names the
line.
"""

import re
from typing import Dict, Iterable

TOKEN = re.compile(r"[\w.\-*+]+")


def read_multfelt(lines: Iterable[str], grdecl: Dict) -> Dict:
    """Consume a MULTFLT section; attach multipliers to existing faults."""
    faults = grdecl.get("FAULTS")
    if not faults:
        raise ValueError(
            "MULTFLT needs the faults it multiplies; read the FAULTS record "
            "first")

    for raw in lines:
        line = raw.strip()
        if line.startswith("/"):
            break
        if not line or line.startswith("--"):
            continue

        fields = TOKEN.findall(line)
        if len(fields) != 2:
            raise ValueError(
                f"a MULTFLT line needs a fault name and one multiplier "
                f"(2 fields), got {len(fields)}: {line!r}")

        name, value = fields
        if name not in faults:
            raise ValueError(
                f"MULTFLT names fault {name!r}, which no FAULTS record "
                f"defines; known faults are {sorted(faults)}")
        try:
            faults[name]["multflt"] = float(value)
        except ValueError:
            raise ValueError(
                f"a MULTFLT multiplier is not a number: {line!r}") from None
    return grdecl


readMultfelt = read_multfelt


if __name__ == "__main__":
    from pymrst.utils.inout.eclipse.deckinput_simple.private.read_faults import (
        read_faults,
    )

    def deck():
        return read_faults(["'F1' 1 1 1 2 1 3 X", "'F2' 4 4 1 2 1 3 Y", "/"])

    def test_it_attaches_a_multiplier() -> None:
        got = read_multfelt(["'F1' 0.001 /"], deck())
        assert got["FAULTS"]["F1"]["multflt"] == 0.001
        print("ok test_it_attaches_a_multiplier")

    def test_faults_without_a_multiplier_are_left_alone() -> None:
        got = read_multfelt(["'F1' 0.001 /"], deck())
        assert "multflt" not in got["FAULTS"]["F2"]
        print("ok test_faults_without_a_multiplier_are_left_alone")

    def test_several_multipliers() -> None:
        got = read_multfelt(["'F1' 0.001", "'F2' 0.5", "/"], deck())
        assert got["FAULTS"]["F1"]["multflt"] == 0.001
        assert got["FAULTS"]["F2"]["multflt"] == 0.5
        print("ok test_several_multipliers")

    def test_it_stops_at_the_terminator() -> None:
        lines = iter(["'F1' 0.001", "/", "PORO"])
        read_multfelt(lines, deck())
        assert next(lines) == "PORO"
        print("ok test_it_stops_at_the_terminator")

    def test_scientific_notation_is_accepted() -> None:
        got = read_multfelt(["'F1' 1.0E-4 /"], deck())
        assert got["FAULTS"]["F1"]["multflt"] == 1.0e-4
        print("ok test_scientific_notation_is_accepted")

    def test_an_unknown_fault_is_rejected() -> None:
        """Almost always a typo; creating the fault would hide it."""
        try:
            read_multfelt(["'F9' 0.001 /"], deck())
        except ValueError as error:
            assert "which no FAULTS record" in str(error)
            print("ok test_an_unknown_fault_is_rejected")
            return
        raise AssertionError("an unknown fault must be rejected")

    def test_multflt_before_faults_is_rejected() -> None:
        try:
            read_multfelt(["'F1' 0.001 /"], {})
        except ValueError as error:
            assert "read the FAULTS record first" in str(error)
            print("ok test_multflt_before_faults_is_rejected")
            return
        raise AssertionError("MULTFLT without FAULTS must be rejected")

    def test_a_wrong_field_count_is_rejected() -> None:
        """MRST's `(~length(split)) == 2` never fires."""
        try:
            read_multfelt(["'F1' 0.001 extra /"], deck())
        except ValueError as error:
            assert "2 fields" in str(error)
            print("ok test_a_wrong_field_count_is_rejected")
            return
        raise AssertionError("a malformed MULTFLT line must be rejected")

    test_it_attaches_a_multiplier()
    test_faults_without_a_multiplier_are_left_alone()
    test_several_multipliers()
    test_it_stops_at_the_terminator()
    test_scientific_notation_is_accepted()
    test_an_unknown_fault_is_rejected()
    test_multflt_before_faults_is_rejected()
    test_a_wrong_field_count_is_rejected()
    print("\nAll tests passed!")
