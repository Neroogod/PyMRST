"""Hysteresis options from EHYSTR (MRST ``props/assignEHYSTR.m``).

EHYSTR turns hysteresis on and says what it applies to. Only its fifth field
is read: 'KR' for relative permeability, 'PC' for capillary pressure, 'BOTH'
for both. The record is also kept whole, because the hysteresis model itself
needs the remaining fields.

This sets flags; it does not implement hysteresis. A deck can therefore carry
EHYSTR and be assigned without the simulation behaving any differently - the
model reads `krHyst` and `pcHyst` and decides.
"""

from typing import Any, Dict, Sequence

# Field 5 of the EHYSTR record, and what each value switches on.
KR_MODES = ("KR", "BOTH")
PC_MODES = ("PC", "BOTH")


def assign_ehystr(f: Dict[str, Any], ehystr: Sequence, reg=None
                  ) -> Dict[str, Any]:
    """Set ``krHyst`` / ``pcHyst`` and keep the record as ``ehystr``."""
    mode = str(ehystr[4]).strip().upper() if len(ehystr) > 4 else ""
    if mode in KR_MODES:
        f["krHyst"] = 1
    if mode in PC_MODES:
        f["pcHyst"] = 1
    f["ehystr"] = ehystr
    return f


assignEHYSTR = assign_ehystr


if __name__ == "__main__":

    def _record(mode):
        return [0.1, 0.0, 1.0, 0.0, mode, "DRAIN", "OIL"]

    def test_kr_sets_only_the_relperm_flag() -> None:
        f = assign_ehystr({}, _record("KR"))
        assert f["krHyst"] == 1 and "pcHyst" not in f
        print("ok test_kr_sets_only_the_relperm_flag")

    def test_pc_sets_only_the_capillary_pressure_flag() -> None:
        f = assign_ehystr({}, _record("PC"))
        assert f["pcHyst"] == 1 and "krHyst" not in f
        print("ok test_pc_sets_only_the_capillary_pressure_flag")

    def test_both_sets_both_flags() -> None:
        f = assign_ehystr({}, _record("BOTH"))
        assert f["krHyst"] == 1 and f["pcHyst"] == 1
        print("ok test_both_sets_both_flags")

    def test_an_unrecognised_mode_sets_nothing() -> None:
        f = assign_ehystr({}, _record("NONE"))
        assert "krHyst" not in f and "pcHyst" not in f
        print("ok test_an_unrecognised_mode_sets_nothing")

    def test_the_whole_record_is_kept() -> None:
        """The hysteresis model needs the fields this function ignores."""
        record = _record("BOTH")
        f = assign_ehystr({}, record)
        assert f["ehystr"] is record
        print("ok test_the_whole_record_is_kept")

    def test_the_mode_is_matched_case_insensitively() -> None:
        assert assign_ehystr({}, _record("both"))["krHyst"] == 1
        assert assign_ehystr({}, _record(" Kr "))["krHyst"] == 1
        print("ok test_the_mode_is_matched_case_insensitively")

    def test_a_short_record_is_tolerated() -> None:
        f = assign_ehystr({}, [0.1, 0.0])
        assert "krHyst" not in f and f["ehystr"] == [0.1, 0.0]
        print("ok test_a_short_record_is_tolerated")

    test_kr_sets_only_the_relperm_flag()
    test_pc_sets_only_the_capillary_pressure_flag()
    test_both_sets_both_flags()
    test_an_unrecognised_mode_sets_nothing()
    test_the_whole_record_is_kept()
    test_the_mode_is_matched_case_insensitively()
    test_a_short_record_is_tolerated()
    print("\nAll tests passed!")
