"""SI unit constants and conversion helpers, ported from MRST `utils/units`."""

from typing import Union

import numpy as np

Quantity = Union[float, np.ndarray]

METER: float = 1.0
SECOND: float = 1.0
KILOGRAM: float = 1.0
KELVIN: float = 1.0
MOL: float = 1.0

PICO: float = 1e-12
NANO: float = 1e-9
MICRO: float = 1e-6
MILLI: float = 1e-3
CENTI: float = 1e-2
DECI: float = 1e-1
KILO: float = 1e3
MEGA: float = 1e6
GIGA: float = 1e9

INCH: float = 2.54 * CENTI * METER
FT: float = 0.3048 * METER

MINUTE: float = 60.0 * SECOND
HOUR: float = 60.0 * MINUTE
DAY: float = 24.0 * HOUR
YEAR: float = 365.2425 * DAY

GRAM: float = 1e-3 * KILOGRAM
POUND: float = 0.45359237 * KILOGRAM
TONNE: float = 1000.0 * KILOGRAM
DALTON: float = 1.66053904020e-27 * KILOGRAM

NEWTON: float = KILOGRAM * METER / SECOND**2
DYNE: float = 1e-5 * NEWTON
LBF: float = POUND * 9.80665 * METER / SECOND**2

PASCAL: float = NEWTON / METER**2
BARSA: float = 1e5 * PASCAL
ATM: float = 101325.0 * PASCAL
PSIA: float = LBF / INCH**2

POISE: float = 0.1 * PASCAL * SECOND
DARCY: float = (CENTI * METER / SECOND) * (CENTI * POISE) / (ATM / (CENTI * METER))

LITRE: float = 1e-3 * METER**3
GALLON: float = 231.0 * INCH**3
STB: float = 42.0 * GALLON

JOULE: float = NEWTON * METER
WATT: float = JOULE / SECOND
BTU: float = 1054.3503 * JOULE

RANKINE: float = 5.0 / 9.0 * KELVIN


# --------------------------------------------------------- electrical units
# MRST sets these to 1: the SI base is already ampere and farad, so the
# constants exist to be written in an expression, not to scale anything.
AMPERE: float = 1.0
FARAD: float = 1.0

# `gal` is the CGS unit of acceleration (1 cm/s^2), not the volume "gallon" -
# which this module also has, as GALLON. Confusing them is off by 5 orders of
# magnitude and MRST's one-word filename does not help.
GAL: float = 0.01 * METER / SECOND ** 2

# ---------------------------------------------------- physical constants
AVOGADRO: float = 6.022140857e23
GAS_CONSTANT: float = 8.3144598                    # J / (mol K)
FARADAY_CONSTANT: float = 96485.33289              # C / mol
VACUUM_PERMITTIVITY: float = 8.854187817e-12       # F / m

# One adsorption site: a mole divided by Avogadro's number.
#
# MRST's `site.m` divides by 6.00221413e23 while its `avogadro.m` states
# 6.022140857e23 - the two disagree in the sixth digit. This uses the single
# AVOGADRO above, so `SITE * AVOGADRO` is exactly 1; MRST's pair gives
# 1.0000000095. The difference is far below any modelling tolerance, but two
# values of a physical constant in one library is worth not reproducing.
SITE: float = 1.0 / AVOGADRO


def convert_from(q: Quantity, unit: float) -> Quantity:
    """Convert `q` from `unit` into SI."""
    return q * unit


def convert_to(q: Quantity, unit: float) -> Quantity:
    """Convert SI quantity `q` into `unit`."""
    return q / unit


if __name__ == "__main__":

    def test_reference_values() -> None:
        assert DARCY == 1e-7 / 101325.0
        assert abs(DARCY - 9.869232667160130e-13) < 5e-28
        assert abs(PSIA - 6894.757293168361) < 1e-11
        assert LBF == 4.4482216152605
        assert abs(STB - 0.158987294928) < 1e-15
        assert YEAR == 31556952.0
        assert DAY == 86400.0
        assert ATM == 101325.0
        assert BARSA == 1e5
        print("ok test_reference_values")

    def test_convert_roundtrip() -> None:
        p_si = convert_from(200.0, BARSA)
        assert p_si == 2e7
        assert convert_to(p_si, BARSA) == 200.0
        assert abs(convert_to(p_si, PSIA) - 2900.7547546) < 1e-5
        print("ok test_convert_roundtrip")

    def test_convert_arrays() -> None:
        perm = convert_from(np.array([1.0, 100.0, 500.0]), MILLI * DARCY)
        assert perm.shape == (3,)
        assert abs(perm[1] - 9.869232667160131e-14) < 1e-28
        assert np.allclose(convert_to(perm, MILLI * DARCY), [1.0, 100.0, 500.0])
        print("ok test_convert_arrays")

    def test_compound_units() -> None:
        rate = convert_from(1000.0, STB / DAY)
        assert abs(rate - 1000.0 * STB / DAY) < 1e-20
        assert convert_from(1.0, CENTI * POISE) == 1e-3
        print("ok test_compound_units")

    def test_the_electrical_units_are_si_base() -> None:
        assert AMPERE == 1.0 and FARAD == 1.0
        print("ok test_the_electrical_units_are_si_base")

    def test_gal_is_an_acceleration_not_a_volume() -> None:
        """`gal` is 1 cm/s^2; `gallon` is a volume. MRST names them alike."""
        assert GAL == 0.01
        assert abs(GALLON / GAL) > 1e-3, (GALLON, GAL)
        assert GALLON != GAL
        print("ok test_gal_is_an_acceleration_not_a_volume")

    def test_a_site_is_one_over_avogadro() -> None:
        """Exactly, unlike MRST, whose two files disagree in the sixth digit."""
        assert SITE * AVOGADRO == 1.0
        print("ok test_a_site_is_one_over_avogadro")

    def test_the_physical_constants_are_codata_values() -> None:
        assert abs(GAS_CONSTANT - 8.3144598) < 1e-7
        assert abs(FARADAY_CONSTANT - 96485.33289) < 1e-5
        assert abs(VACUUM_PERMITTIVITY - 8.854187817e-12) < 1e-20
        print("ok test_the_physical_constants_are_codata_values")

    def test_the_gas_constant_relates_boltzmann_and_avogadro() -> None:
        """R = k_B * N_A, an independent check on two of the constants."""
        boltzmann = 1.38064852e-23
        assert abs(GAS_CONSTANT - boltzmann * AVOGADRO) < 1e-6, \
            GAS_CONSTANT - boltzmann * AVOGADRO
        print("ok test_the_gas_constant_relates_boltzmann_and_avogadro")

    test_the_electrical_units_are_si_base()
    test_gal_is_an_acceleration_not_a_volume()
    test_a_site_is_one_over_avogadro()
    test_the_physical_constants_are_codata_values()
    test_the_gas_constant_relates_boltzmann_and_avogadro()
    test_reference_values()
    test_convert_roundtrip()
    test_convert_arrays()
    test_compound_units()
    print("\nAll tests passed!")

