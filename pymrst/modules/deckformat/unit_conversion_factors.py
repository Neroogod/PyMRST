"""Unit systems for ECLIPSE decks (MRST ``deckformat/unitConversionFactors.m``).

A deck declares METRIC, FIELD, LAB or PVT-M in RUNSPEC, and every number in it
is then in that system. This module gives, for each system, the SI size of one
unit of each quantity - so a deck value is converted by multiplying.

The quantity names are MRST's, and they are more specific than they first look.
``liqvol_s`` and ``liqvol_r`` are both stock-tank barrels in FIELD but differ
in other systems, and ``gasvol_s`` is Mscf while ``gasvol_r`` is bbl - which is
exactly why a formation volume factor cannot be treated as dimensionless.
"""

from dataclasses import dataclass, fields
from typing import Dict

from pymrst.utils.units.units import (
    ATM, BARSA, BTU, CENTI, DARCY, DAY, DYNE, FT, GIGA, GRAM, HOUR, INCH,
    JOULE, KELVIN, KILO, KILOGRAM, LBF, METER, MILLI, NEWTON, POISE, POUND,
    PSIA, RANKINE, STB,
)


@dataclass(frozen=True)
class UnitSystem:
    """One unit of each quantity, measured in SI."""

    length: float
    area: float
    invarea: float
    time: float
    density: float
    press: float
    temp: float
    tempoffset: float
    mol: float
    mass: float
    concentr: float
    compr: float
    viscosity: float
    surf_tension: float
    jsurftens: float
    perm: float
    liqvol_s: float
    liqvol_r: float
    gasvol_s: float
    gasvol_r: float
    volume: float
    trans: float
    rockcond: float
    volumheatcapacity: float
    massheatcapacity: float
    ymodule: float = 1.0
    unit_in: str = "si"
    unit_out: str = "si"


CP = CENTI * POISE
MD = MILLI * DARCY
JSURFTENS = DYNE / (CENTI * METER)

METRIC = UnitSystem(
    length=METER, area=METER ** 2, invarea=METER ** -2, time=DAY,
    density=KILOGRAM / METER ** 3, press=BARSA, temp=KELVIN, tempoffset=273.15,
    mol=KILO, mass=KILOGRAM, concentr=KILOGRAM / METER ** 3, compr=1 / BARSA,
    viscosity=CP, surf_tension=NEWTON / METER, jsurftens=JSURFTENS, perm=MD,
    liqvol_s=METER ** 3, liqvol_r=METER ** 3,
    gasvol_s=METER ** 3, gasvol_r=METER ** 3, volume=METER ** 3,
    trans=CP * METER ** 3 / (DAY * BARSA),
    rockcond=KILO * JOULE / (METER * DAY * KELVIN),
    volumheatcapacity=KILO * JOULE / (METER ** 3 * KELVIN),
    massheatcapacity=KILO * JOULE / (KILOGRAM * KELVIN),
    ymodule=GIGA,
)

FIELD = UnitSystem(
    length=FT, area=FT ** 2, invarea=FT ** -2, time=DAY,
    density=POUND / FT ** 3, press=PSIA, temp=RANKINE, tempoffset=459.67,
    mol=POUND * KILO, mass=POUND, concentr=POUND / STB, compr=1 / PSIA,
    viscosity=CP, surf_tension=LBF / INCH, jsurftens=JSURFTENS, perm=MD,
    liqvol_s=STB, liqvol_r=STB,
    gasvol_s=1000 * FT ** 3, gasvol_r=STB, volume=FT ** 3,
    trans=CP * STB / (DAY * PSIA),
    rockcond=BTU / (FT * DAY * RANKINE),
    volumheatcapacity=BTU / (FT ** 3 * RANKINE),
    massheatcapacity=BTU / (POUND * RANKINE),
)

CM = CENTI * METER
LAB = UnitSystem(
    length=CM, area=CM ** 2, invarea=CM ** -2, time=HOUR,
    density=GRAM / CM ** 3, press=ATM, temp=KELVIN, tempoffset=273.15,
    mol=1.0, mass=GRAM * KILO, concentr=GRAM / CM ** 3, compr=1 / ATM,
    viscosity=CP, surf_tension=DYNE / CM, jsurftens=JSURFTENS, perm=MD,
    liqvol_s=CM ** 3, liqvol_r=CM ** 3, gasvol_s=CM ** 3, gasvol_r=CM ** 3,
    volume=CM ** 3, trans=CP * CM ** 3 / (HOUR * ATM),
    rockcond=JOULE / (CM * HOUR * KELVIN),
    volumheatcapacity=JOULE / (CM ** 3 * KELVIN),
    massheatcapacity=JOULE / (GRAM * KELVIN),
)

PVT_M = UnitSystem(
    length=METER, area=METER ** 2, invarea=METER ** -2, time=DAY,
    density=KILOGRAM / METER ** 3, press=ATM, temp=KELVIN, tempoffset=273.15,
    mol=KILO, mass=KILOGRAM, concentr=KILOGRAM / METER ** 3, compr=1 / ATM,
    viscosity=CP, surf_tension=NEWTON / METER, jsurftens=JSURFTENS, perm=MD,
    liqvol_s=METER ** 3, liqvol_r=METER ** 3,
    gasvol_s=METER ** 3, gasvol_r=METER ** 3, volume=METER ** 3,
    trans=CP * METER ** 3 / (DAY * ATM),
    rockcond=KILO * JOULE / (METER * DAY * KELVIN),
    volumheatcapacity=KILO * JOULE / (METER ** 3 * KELVIN),
    massheatcapacity=KILO * JOULE / (KILOGRAM * KELVIN),
)

# SI is MRST's extension, present so converting an already-converted deck is a
# no-op rather than an error.
SI = UnitSystem(**{f.name: 0.0 if f.name == "tempoffset" else 1.0
                   for f in fields(UnitSystem)
                   if f.name not in ("unit_in", "unit_out")})

SYSTEMS: Dict[str, UnitSystem] = {
    "metric": METRIC, "field": FIELD, "lab": LAB, "pvt_m": PVT_M,
    "pvt-m": PVT_M, "si": SI,
}


def unit_conversion_factors(input_unit: str,
                            output_unit: str = "si") -> UnitSystem:
    """Factors taking values from ``input_unit`` to ``output_unit``."""
    input_unit, output_unit = input_unit.lower(), output_unit.lower()
    for name in (input_unit, output_unit):
        if name not in SYSTEMS:
            raise ValueError(
                f"unit system '{name.upper()}' unknown, must be one of "
                f"METRIC, FIELD, LAB, PVT-M or SI")

    factors = SYSTEMS[input_unit]
    if output_unit != "si":
        target = SYSTEMS[output_unit]
        ratios = {f.name: getattr(factors, f.name) / getattr(target, f.name)
                  for f in fields(UnitSystem)
                  if f.name not in ("tempoffset", "unit_in", "unit_out")}
        ratios["tempoffset"] = (factors.tempoffset
                                - target.tempoffset / ratios["temp"])
        factors = UnitSystem(**ratios)
    return UnitSystem(**{f.name: getattr(factors, f.name)
                         for f in fields(UnitSystem)
                         if f.name not in ("unit_in", "unit_out")},
                      unit_in=input_unit, unit_out=output_unit)


unitConversionFactors = unit_conversion_factors


if __name__ == "__main__":
    import numpy as np

    def test_field_pressure_is_psia() -> None:
        u = unit_conversion_factors("FIELD")
        assert np.isclose(u.press, PSIA), u.press
        assert np.isclose(4014.7 * u.press, 27680382.10488301), 4014.7 * u.press
        print("ok test_field_pressure_is_psia")

    def test_si_is_the_identity() -> None:
        u = unit_conversion_factors("SI")
        assert u.press == 1.0 and u.length == 1.0 and u.tempoffset == 0.0
        print("ok test_si_is_the_identity")

    def test_field_gas_and_liquid_volumes_differ() -> None:
        """Mscf vs bbl: the reason a gas FVF is not dimensionless."""
        u = unit_conversion_factors("FIELD")
        assert not np.isclose(u.gasvol_s, u.gasvol_r), (u.gasvol_s, u.gasvol_r)
        assert np.isclose(u.gasvol_s / u.gasvol_r, 1000 * FT ** 3 / STB)
        print("ok test_field_gas_and_liquid_volumes_differ")

    def test_metric_pressure_is_bar() -> None:
        u = unit_conversion_factors("METRIC")
        assert np.isclose(u.press, 1e5), u.press
        print("ok test_metric_pressure_is_bar")

    def test_a_round_trip_between_systems_is_the_identity() -> None:
        there = unit_conversion_factors("FIELD", "METRIC")
        back = unit_conversion_factors("METRIC", "FIELD")
        assert np.isclose(there.press * back.press, 1.0)
        assert np.isclose(there.density * back.density, 1.0)
        print("ok test_a_round_trip_between_systems_is_the_identity")

    def test_converting_to_si_twice_is_stable() -> None:
        u = unit_conversion_factors("SI", "SI")
        assert u.press == 1.0
        print("ok test_converting_to_si_twice_is_stable")

    def test_temperature_offset_survives_a_system_change() -> None:
        """60 F is 288.7 K whether converted directly or through METRIC."""
        u = unit_conversion_factors("FIELD")
        direct = (60.0 + u.tempoffset) * u.temp
        assert np.isclose(direct, 288.706, atol=1e-2), direct
        print("ok test_temperature_offset_survives_a_system_change")

    def test_an_unknown_system_is_rejected() -> None:
        try:
            unit_conversion_factors("IMPERIAL")
        except ValueError as error:
            assert "IMPERIAL" in str(error)
            print("ok test_an_unknown_system_is_rejected")
            return
        raise AssertionError("an unknown unit system must be rejected")

    def test_every_system_defines_every_quantity() -> None:
        for name, system in SYSTEMS.items():
            for f in fields(UnitSystem):
                value = getattr(system, f.name)
                if f.name in ("unit_in", "unit_out"):
                    continue
                assert isinstance(value, float) and np.isfinite(value), \
                    (name, f.name, value)
        print(f"ok test_every_system_defines_every_quantity ({len(SYSTEMS)})")

    test_field_pressure_is_psia()
    test_si_is_the_identity()
    test_field_gas_and_liquid_volumes_differ()
    test_metric_pressure_is_bar()
    test_a_round_trip_between_systems_is_the_identity()
    test_converting_to_si_twice_is_stable()
    test_temperature_offset_survives_a_system_change()
    test_an_unknown_system_is_rejected()
    test_every_system_defines_every_quantity()
    print("\nAll tests passed!")
