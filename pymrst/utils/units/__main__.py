"""Package self-tests for `pymrst.utils.units`."""

import numpy as np

from pymrst.utils.units import (
    ATM,
    BARSA,
    CENTI,
    DARCY,
    DAY,
    LBF,
    MILLI,
    POISE,
    PSIA,
    STB,
    YEAR,
    convert_from,
    convert_to,
)


def test_reference_values() -> None:
    """Constants match the values MRST 2024a produces."""
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
    """`convert_to` inverts `convert_from` for scalars."""
    p_si = convert_from(200.0, BARSA)
    assert p_si == 2e7
    assert convert_to(p_si, BARSA) == 200.0
    assert abs(convert_to(p_si, PSIA) - 2900.7547546) < 1e-5
    print("ok test_convert_roundtrip")


def test_convert_arrays() -> None:
    """Conversions broadcast over NumPy arrays."""
    perm = convert_from(np.array([1.0, 100.0, 500.0]), MILLI * DARCY)
    assert perm.shape == (3,)
    assert abs(perm[1] - 9.869232667160131e-14) < 1e-28
    assert np.allclose(convert_to(perm, MILLI * DARCY), [1.0, 100.0, 500.0])
    print("ok test_convert_arrays")


def test_compound_units() -> None:
    """Compound unit expressions read like MRST."""
    rate = convert_from(1000.0, STB / DAY)
    assert abs(rate - 1000.0 * STB / DAY) < 1e-20
    assert convert_from(1.0, CENTI * POISE) == 1e-3
    print("ok test_compound_units")


test_reference_values()
test_convert_roundtrip()
test_convert_arrays()
test_compound_units()
print("\nAll tests passed!")

