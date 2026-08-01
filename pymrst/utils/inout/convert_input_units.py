"""Convert a deck's arrays to SI (MRST ``utils/inout/convertInputUnits.m``).

A GRDECL file carries lengths, permeabilities and transmissibilities in
whatever system it was written in - METRIC, FIELD, LAB. PyMRST works in SI
throughout, so this is the step between reading a deck and using it. Pass the
unit system from `get_unit_system`.

**It refuses to guess.** A keyword it does not recognise raises, rather than
passing the array through unconverted - which is MRST's behaviour and the right
one: silently leaving a permeability in millidarcy is a factor-of-1e-15 error
that will not announce itself. The cost is that a deck with an unusual keyword
stops here, so `known_only=True` is offered for the case where you would rather
convert what is understood and be told the rest.

**`GRIDUNIT` and `MAPUNITS` override the system for lengths.** A deck can be
declared FIELD while stating its grid is in metres, and then `COORD` and
`ZCORN` are metres regardless. Both records are honoured, as MRST does; an
unrecognised one warns and falls back to metres rather than failing, since the
grid is usually still usable.

Not every keyword needs converting - `PORO`, `ACTNUM`, `SATNUM` and the
multipliers are dimensionless - and those are listed explicitly rather than
defaulted, so a new keyword lands in the unknown branch and gets noticed.
"""

import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from pymrst.utils.units.get_unit_system import UnitSystem, get_unit_system

PERMEABILITY = {"PERMX", "PERMXY", "PERMXZ", "PERMYX", "PERMY", "PERMYZ",
                "PERMZX", "PERMZY", "PERMZ", "PERMH"}
LENGTH = {"DXV", "DYV", "DZV", "DEPTHZ", "COORD", "ZCORN"}
TRANSMISSIBILITY = {"TRANX", "TRANY", "TRANZ"}
DIMENSIONLESS = {
    "cartDims", "UnhandledKeywords", "ROCKTYPE", "ACTNUM", "NTG", "PORO",
    "SATNUM", "FAULTS", "MULTFLT", "MULTX", "MULTX_", "MULTY", "MULTY_",
    "MULTZ", "MULTZ_", "VSH", "GDORIENT", "MAPUNITS", "GRIDUNIT",
}

_NAMED_LENGTHS = {
    "metres": 1.0, "meters": 1.0, "metre": 1.0, "meter": 1.0, "m": 1.0,
    "foot": 0.3048, "feet": 0.3048, "ft": 0.3048,
    "centimetre": 0.01, "centimeter": 0.01, "cm": 0.01,
}


def _declared_length(record, default: float) -> float:
    """The length unit named by a GRIDUNIT/MAPUNITS record, if any."""
    if record is None:
        return default
    if isinstance(record, (list, tuple, np.ndarray)):
        if len(record) == 0:
            return default
        record = record[0]
    if not isinstance(record, str):
        return default
    key = record.strip().lower()
    if key in _NAMED_LENGTHS:
        return _NAMED_LENGTHS[key]
    warnings.warn(f"unsupported unit {record!r}; assuming metres",
                  stacklevel=3)
    return 1.0


def convert_input_units(grdecl: Dict, units, exclude: Sequence[str] = (),
                        known_only: bool = False) -> Tuple[Dict, List[str]]:
    """Return ``(converted_deck, untouched_keywords)``, values in SI."""
    if isinstance(units, str):
        units = get_unit_system(units)
    if not isinstance(units, UnitSystem):
        raise TypeError(
            f"units must be a UnitSystem or a system name, got {type(units).__name__}")

    exclude = set(exclude)
    out = dict(grdecl)
    grid_length = _declared_length(grdecl.get("GRIDUNIT"), units.length)
    map_length = _declared_length(grdecl.get("MAPUNITS"), units.length)

    untouched: List[str] = []
    for key, value in grdecl.items():
        if key in exclude or key in DIMENSIONLESS or value is None:
            if key in exclude:
                untouched.append(key)
            continue
        if key in PERMEABILITY:
            factor = units.perm
        elif key in LENGTH:
            factor = grid_length
        elif key in TRANSMISSIBILITY:
            factor = units.trans
        elif key == "MAPAXES":
            factor = map_length
        elif known_only:
            untouched.append(key)
            continue
        else:
            raise ValueError(
                f"no unit conversion is known for {key!r}; add it to this "
                f"module's tables, list it in `exclude`, or pass "
                f"known_only=True to convert the rest and be told about it")
        out[key] = np.asarray(value, dtype=float) * factor
    return out, sorted(untouched)


convertInputUnits = convert_input_units


if __name__ == "__main__":

    def deck():
        return {"cartDims": np.array([2, 2, 2]),
                "COORD": np.ones(54), "ZCORN": np.ones(64),
                "PERMX": np.full(8, 100.0), "PORO": np.full(8, 0.2)}

    def test_metric_converts_permeability_from_millidarcy() -> None:
        got, _ = convert_input_units(deck(), "metric")
        assert np.allclose(got["PERMX"], 100.0 * 9.869232667160131e-16), \
            got["PERMX"][0]
        print("ok test_metric_converts_permeability_from_millidarcy")

    def test_field_converts_lengths_from_feet() -> None:
        got, _ = convert_input_units(deck(), "field")
        assert np.allclose(got["COORD"], 0.3048)
        assert np.allclose(got["ZCORN"], 0.3048)
        print("ok test_field_converts_lengths_from_feet")

    def test_si_changes_nothing() -> None:
        got, _ = convert_input_units(deck(), "si")
        for key in ("COORD", "ZCORN", "PERMX"):
            assert np.allclose(got[key], deck()[key]), key
        print("ok test_si_changes_nothing")

    def test_dimensionless_arrays_are_untouched() -> None:
        got, _ = convert_input_units(deck(), "field")
        assert np.allclose(got["PORO"], 0.2)
        assert np.array_equal(got["cartDims"], [2, 2, 2])
        print("ok test_dimensionless_arrays_are_untouched")

    def test_gridunit_overrides_the_system() -> None:
        """A FIELD deck can still state its grid is in metres."""
        d = deck()
        d["GRIDUNIT"] = ["METRES"]
        got, _ = convert_input_units(d, "field")
        assert np.allclose(got["COORD"], 1.0), "GRIDUNIT should win"
        # Permeability is unaffected by GRIDUNIT.
        assert np.allclose(got["PERMX"], 100.0 * 9.869232667160131e-16)
        print("ok test_gridunit_overrides_the_system")

    def test_an_unknown_gridunit_warns_and_assumes_metres() -> None:
        d = deck()
        d["GRIDUNIT"] = ["furlongs"]
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            got, _ = convert_input_units(d, "field")
        assert any("furlongs" in str(w.message) for w in caught), caught
        assert np.allclose(got["COORD"], 1.0)
        print("ok test_an_unknown_gridunit_warns_and_assumes_metres")

    def test_an_unknown_keyword_is_rejected() -> None:
        """Silently passing it through would be a factor-of-1e15 error."""
        d = deck()
        d["WIBBLE"] = np.ones(8)
        try:
            convert_input_units(d, "metric")
        except ValueError as error:
            assert "no unit conversion is known for 'WIBBLE'" in str(error)
            print("ok test_an_unknown_keyword_is_rejected")
            return
        raise AssertionError("an unknown keyword must be rejected")

    def test_known_only_converts_the_rest_and_reports() -> None:
        d = deck()
        d["WIBBLE"] = np.ones(8)
        got, untouched = convert_input_units(d, "metric", known_only=True)
        assert untouched == ["WIBBLE"]
        assert np.allclose(got["WIBBLE"], 1.0), "should be passed through as-is"
        assert not np.allclose(got["PERMX"], 100.0), "the rest still converts"
        print("ok test_known_only_converts_the_rest_and_reports")

    def test_exclude_skips_a_keyword() -> None:
        got, untouched = convert_input_units(deck(), "field", exclude=["COORD"])
        assert np.allclose(got["COORD"], 1.0)
        assert np.allclose(got["ZCORN"], 0.3048)
        assert untouched == ["COORD"]
        print("ok test_exclude_skips_a_keyword")

    def test_the_input_is_not_modified() -> None:
        d = deck()
        convert_input_units(d, "field")
        assert np.allclose(d["COORD"], 1.0)
        print("ok test_the_input_is_not_modified")

    def test_a_unit_system_object_works_too() -> None:
        from pymrst.utils.units.get_unit_system import get_unit_system as gus
        assert np.allclose(convert_input_units(deck(), gus("field"))[0]["COORD"],
                           convert_input_units(deck(), "field")[0]["COORD"])
        print("ok test_a_unit_system_object_works_too")

    def test_a_bad_units_argument_is_rejected() -> None:
        try:
            convert_input_units(deck(), 42)
        except TypeError as error:
            assert "UnitSystem" in str(error)
            print("ok test_a_bad_units_argument_is_rejected")
            return
        raise AssertionError("a nonsense unit system must be rejected")

    test_metric_converts_permeability_from_millidarcy()
    test_field_converts_lengths_from_feet()
    test_si_changes_nothing()
    test_dimensionless_arrays_are_untouched()
    test_gridunit_overrides_the_system()
    test_an_unknown_gridunit_warns_and_assumes_metres()
    test_an_unknown_keyword_is_rejected()
    test_known_only_converts_the_rest_and_reports()
    test_exclude_skips_a_keyword()
    test_the_input_is_not_modified()
    test_a_unit_system_object_works_too()
    test_a_bad_units_argument_is_rejected()
    print("\nAll tests passed!")
