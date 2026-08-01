"""Read an EQUIL record (MRST ``.../deckinput_simple/private/readEQUIL.m``).

EQUIL states the equilibration data for each region: a datum depth and
pressure, then the water-oil and gas-oil contacts with their capillary
pressures, then table selectors. Eleven items, all defaulting to 0 except the
tenth, which defaults to 1.

Returned in **SI**: the odd items (1st, 3rd, 5th) are depths and the even ones
(2nd, 4th, 6th) are pressures, converted from the deck's system. That
alternation is why the conversion looks arbitrary - it follows EQUIL's own
column layout:

    1 datum depth   2 datum pressure
    3 OWC depth     4 OWC capillary pressure
    5 GOC depth     6 GOC capillary pressure

Items 7-11 are integers selecting tables and are left alone.

**`metric` selects the unit system**, as in MRST: metres and bar, or feet and
psi. It is a boolean there; here it also accepts the system name, since a
caller that has read `GRIDUNIT` has a string rather than a flag.
"""

from typing import Sequence, Union

import numpy as np

from pymrst.utils.inout.eclipse.deckinput_simple.read_defaulted_record import (
    DeckCursor,
    read_defaulted_kw,
)
from pymrst.utils.units.units import BARSA, FT, METER, PSIA

TEMPLATE = ["0"] * 11
TEMPLATE[9] = "1"

DEPTH_ITEMS = (0, 2, 4)
PRESSURE_ITEMS = (1, 3, 5)


def _system(metric: Union[bool, str]):
    """(depth unit, pressure unit) for a flag or a system name."""
    if isinstance(metric, str):
        name = metric.strip().lower()
        if name not in ("metric", "field"):
            raise ValueError(
                f"EQUIL units must be 'metric' or 'field', got {metric!r}")
        metric = name == "metric"
    return (METER, BARSA) if metric else (FT, PSIA)


def read_equil(tokens: Sequence[str], ntequil: int = 1,
               metric: Union[bool, str] = True) -> np.ndarray:
    """Return ``(ntequil, 11)`` equilibration data, in SI."""
    if ntequil < 1:
        raise ValueError(f"there must be at least one EQUIL region, got {ntequil}")
    depth_unit, pressure_unit = _system(metric)

    cursor = tokens if isinstance(tokens, DeckCursor) else DeckCursor(list(tokens))
    records = read_defaulted_kw(cursor, TEMPLATE, n_rec=ntequil)

    equil = np.array([[float(item) for item in record] for record in records],
                     dtype=float)
    if equil.shape != (ntequil, 11):
        raise ValueError(
            f"expected {ntequil} EQUIL records of 11 items, got "
            f"{equil.shape[0]} of {equil.shape[1] if equil.ndim > 1 else 0}")

    equil[:, list(DEPTH_ITEMS)] *= depth_unit
    equil[:, list(PRESSURE_ITEMS)] *= pressure_unit
    return equil


readEQUIL = read_equil


if __name__ == "__main__":

    def tokens(text):
        return text.replace("/", " / ").split()

    def test_a_full_record_in_metric() -> None:
        got = read_equil(tokens("2000 300 2200 0 1800 0 1 1 0 1 0 /"))
        assert got.shape == (1, 11)
        assert np.isclose(got[0, 0], 2000.0)             # metres
        assert np.isclose(got[0, 1], 300.0 * BARSA)      # bar -> Pa
        assert np.isclose(got[0, 2], 2200.0)
        print("ok test_a_full_record_in_metric")

    def test_field_units_convert_depth_and_pressure() -> None:
        got = read_equil(tokens("2000 300 2200 0 1800 0 1 1 0 1 0 /"),
                         metric=False)
        assert np.isclose(got[0, 0], 2000.0 * FT)
        assert np.isclose(got[0, 1], 300.0 * PSIA)
        print("ok test_field_units_convert_depth_and_pressure")

    def test_the_table_selectors_are_not_converted() -> None:
        """Items 7-11 are integers, not physical quantities."""
        got = read_equil(tokens("2000 300 2200 0 1800 0 3 4 5 6 7 /"))
        assert got[0, 6:].tolist() == [3.0, 4.0, 5.0, 6.0, 7.0]
        print("ok test_the_table_selectors_are_not_converted")

    def test_defaults_are_zero_except_the_tenth() -> None:
        got = read_equil(tokens("2000 300 9* /"))
        assert np.isclose(got[0, 0], 2000.0)
        assert np.isclose(got[0, 1], 300.0 * BARSA)
        assert got[0, 2:9].tolist() == [0.0] * 7
        assert got[0, 9] == 1.0, "item 10 defaults to 1"
        assert got[0, 10] == 0.0
        print("ok test_defaults_are_zero_except_the_tenth")

    def test_several_regions() -> None:
        got = read_equil(tokens("2000 300 9* / 2500 350 9* /"), ntequil=2)
        assert got.shape == (2, 11)
        assert np.isclose(got[1, 0], 2500.0)
        assert np.isclose(got[1, 1], 350.0 * BARSA)
        print("ok test_several_regions")

    def test_a_system_name_works_as_well_as_a_flag() -> None:
        by_name = read_equil(tokens("2000 300 9* /"), metric="field")
        by_flag = read_equil(tokens("2000 300 9* /"), metric=False)
        assert np.allclose(by_name, by_flag)
        print("ok test_a_system_name_works_as_well_as_a_flag")

    def test_metric_depth_is_the_identity() -> None:
        got = read_equil(tokens("1234.5 1 9* /"))
        assert np.isclose(got[0, 0], 1234.5), "metres are already SI"
        print("ok test_metric_depth_is_the_identity")

    def test_a_bad_system_name_is_rejected() -> None:
        try:
            read_equil(tokens("2000 300 9* /"), metric="lab")
        except ValueError as error:
            assert "'metric' or 'field'" in str(error)
            print("ok test_a_bad_system_name_is_rejected")
            return
        raise AssertionError("an unknown unit system must be rejected")

    def test_zero_regions_is_rejected() -> None:
        try:
            read_equil(tokens("2000 300 9* /"), ntequil=0)
        except ValueError as error:
            assert "at least one EQUIL region" in str(error)
            print("ok test_zero_regions_is_rejected")
            return
        raise AssertionError("ntequil must be at least 1")

    test_a_full_record_in_metric()
    test_field_units_convert_depth_and_pressure()
    test_the_table_selectors_are_not_converted()
    test_defaults_are_zero_except_the_tenth()
    test_several_regions()
    test_a_system_name_works_as_well_as_a_flag()
    test_metric_depth_is_the_identity()
    test_a_bad_system_name_is_rejected()
    test_zero_regions_is_rejected()
    print("\nAll tests passed!")
