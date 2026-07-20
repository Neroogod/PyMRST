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

    test_reference_values()
    test_convert_roundtrip()
    test_convert_arrays()
    test_compound_units()
    print("\nAll tests passed!")

