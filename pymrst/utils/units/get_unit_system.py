"""Named unit systems (MRST ``utils/units/getUnitSystem.m``).

Seven quantities - length, time, pressure, viscosity, permeability, reservoir
volume and transmissibility - for each of METRIC, FIELD, LAB and SI. Each value
is the SI size of one unit in that system, so a value written in the system is
converted by multiplying.

**This is not `deckformat`'s `unit_conversion_factors`, and the two should stay
separate.** That one carries 25 quantities, knows about temperature offsets,
and converts *between* two systems for reading ECLIPSE decks. This one names
four systems with seven fields each and is what `convertInputUnits` uses.
Merging them would give deck reading a vocabulary it does not need and this a
complexity it does not want; where they overlap - length, pressure, viscosity,
permeability - they agree, and the tests below check that rather than assume
it.
"""

from dataclasses import dataclass, fields

from pymrst.utils.units.units import (
    ATM,
    BARSA,
    CENTI,
    DARCY,
    DAY,
    FT,
    HOUR,
    METER,
    MILLI,
    POISE,
    PSIA,
    STB,
)

SYSTEMS = ("metric", "field", "lab", "si")


@dataclass(frozen=True)
class UnitSystem:
    """One unit of each quantity, measured in SI."""

    length: float
    time: float
    press: float
    viscosity: float
    perm: float
    resvolume: float
    trans: float


_CP = CENTI * POISE
_CM = CENTI * METER

_METRIC = UnitSystem(
    length=METER, time=DAY, press=BARSA, viscosity=_CP, perm=MILLI * DARCY,
    resvolume=METER ** 3, trans=_CP * METER ** 3 / (DAY * BARSA))
_FIELD = UnitSystem(
    length=FT, time=DAY, press=PSIA, viscosity=_CP, perm=MILLI * DARCY,
    resvolume=STB, trans=_CP * STB / (DAY * PSIA))
_LAB = UnitSystem(
    length=_CM, time=HOUR, press=ATM, viscosity=_CP, perm=MILLI * DARCY,
    resvolume=_CM ** 3, trans=_CP * _CM ** 3 / (HOUR * ATM))
_SI = UnitSystem(length=1.0, time=1.0, press=1.0, viscosity=1.0, perm=1.0,
                 resvolume=1.0, trans=1.0)

_BY_NAME = {"metric": _METRIC, "field": _FIELD, "lab": _LAB, "si": _SI}


def get_unit_system(name: str) -> UnitSystem:
    """The unit system called ``name``; one of METRIC, FIELD, LAB, SI."""
    try:
        return _BY_NAME[name.lower()]
    except (KeyError, AttributeError):
        raise ValueError(
            f"input unit system must be METRIC, FIELD, LAB or SI, "
            f"not {name!r}") from None


getUnitSystem = get_unit_system


if __name__ == "__main__":
    import numpy as np

    def test_every_system_defines_every_quantity() -> None:
        for name in SYSTEMS:
            system = get_unit_system(name)
            for field in fields(UnitSystem):
                value = getattr(system, field.name)
                assert isinstance(value, float) and value > 0, (name, field.name)
        print(f"ok test_every_system_defines_every_quantity ({len(SYSTEMS)})")

    def test_si_is_all_ones() -> None:
        system = get_unit_system("si")
        assert all(getattr(system, f.name) == 1.0 for f in fields(UnitSystem))
        print("ok test_si_is_all_ones")

    def test_the_name_is_case_insensitive() -> None:
        assert get_unit_system("METRIC") is get_unit_system("metric")
        print("ok test_the_name_is_case_insensitive")

    def test_field_lengths_are_feet_and_metric_are_metres() -> None:
        assert np.isclose(get_unit_system("field").length, 0.3048)
        assert get_unit_system("metric").length == 1.0
        print("ok test_field_lengths_are_feet_and_metric_are_metres")

    def test_transmissibility_is_consistent_with_its_parts() -> None:
        """trans = viscosity * volume / (time * pressure), by definition."""
        for name in SYSTEMS:
            system = get_unit_system(name)
            expected = (system.viscosity * system.resvolume
                        / (system.time * system.press))
            assert np.isclose(system.trans, expected, rtol=1e-12), name
        print("ok test_transmissibility_is_consistent_with_its_parts")

    def test_it_agrees_with_the_deck_conversion_factors() -> None:
        """The two unit tables overlap; where they do, they must not differ.

        `deckformat`'s table is the richer one - 25 quantities, temperature
        offsets, conversion between systems - and this is the older, smaller
        one. They are kept separate on purpose, so this checks the shared
        fields rather than trusting that they were written from the same source.
        """
        from pymrst.modules.deckformat.unit_conversion_factors import (
            unit_conversion_factors,
        )
        for name in ("metric", "field", "lab"):
            mine = get_unit_system(name)
            theirs = unit_conversion_factors(name)
            for shared in ("length", "press", "viscosity", "perm"):
                assert np.isclose(getattr(mine, shared),
                                  getattr(theirs, shared), rtol=1e-12), \
                    (name, shared, getattr(mine, shared), getattr(theirs, shared))
        print("ok test_it_agrees_with_the_deck_conversion_factors")

    def test_the_two_tables_are_not_the_same_object() -> None:
        """Stated so nobody 'deduplicates' them; see the module docstring."""
        from pymrst.modules.deckformat.unit_conversion_factors import (
            unit_conversion_factors,
        )
        assert not hasattr(get_unit_system("metric"), "tempoffset")
        assert hasattr(unit_conversion_factors("metric"), "tempoffset")
        print("ok test_the_two_tables_are_not_the_same_object")

    def test_an_unknown_system_is_rejected() -> None:
        try:
            get_unit_system("imperial")
        except ValueError as error:
            assert "METRIC, FIELD, LAB or SI" in str(error)
            print("ok test_an_unknown_system_is_rejected")
            return
        raise AssertionError("an unknown unit system must be rejected")

    test_every_system_defines_every_quantity()
    test_si_is_all_ones()
    test_the_name_is_case_insensitive()
    test_field_lengths_are_feet_and_metric_are_metres()
    test_transmissibility_is_consistent_with_its_parts()
    test_it_agrees_with_the_deck_conversion_factors()
    test_the_two_tables_are_not_the_same_object()
    test_an_unknown_system_is_rejected()
    print("\nAll tests passed!")
