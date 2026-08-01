"""Dead-oil PVT from PVDO (MRST ``ad-props/props/assignPVDO.m``).

PVDO describes oil with no dissolved gas: B and viscosity depend on pressure
alone, so both are straight table lookups with no saturation branch.

The table stores B, but every consumer wants b = 1/B. The reciprocal is taken
here, once, on the tabulated values - not inside the returned function, where
it would be recomputed on every call and, worse, would interpolate B and then
invert rather than interpolating b. Those are different functions.
"""

from typing import Any, Callable, Dict, List

import numpy as np


def assign_pvdo(f: Dict[str, Any], pvdo, reg) -> Dict[str, Any]:
    """Store ``bO`` and ``muO`` as one interpolant per PVT region."""
    b_functions: List[Callable] = []
    mu_functions: List[Callable] = []

    for index in range(reg.pvt):
        table = np.asarray(pvdo[index], dtype=float)
        pressure, b, viscosity = table[:, 0], 1.0 / table[:, 1], table[:, 2]
        b_functions.append(
            lambda p, *, x=pressure, y=b: reg.interp(x, y, p))
        mu_functions.append(
            lambda p, *, x=pressure, y=viscosity: reg.interp(x, y, p))

    f["bO"], f["muO"] = b_functions, mu_functions
    return f


assignPVDO = assign_pvdo


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    TABLE = np.array([[1.0e7, 1.20, 1.0e-3],
                      [2.0e7, 1.15, 1.1e-3],
                      [3.0e7, 1.10, 1.2e-3]])

    def test_b_is_the_reciprocal_of_the_tabulated_fvf() -> None:
        f = assign_pvdo({}, [TABLE], Regions())
        assert np.isclose(f["bO"][0](2.0e7), 1.0 / 1.15)
        print("ok test_b_is_the_reciprocal_of_the_tabulated_fvf")

    def test_viscosity_is_read_straight_from_the_table() -> None:
        f = assign_pvdo({}, [TABLE], Regions())
        assert np.isclose(f["muO"][0](3.0e7), 1.2e-3)
        print("ok test_viscosity_is_read_straight_from_the_table")

    def test_b_interpolates_rather_than_inverting_an_interpolated_fvf() -> None:
        """1/interp(B) and interp(1/B) differ; MRST tabulates the reciprocal."""
        f = assign_pvdo({}, [TABLE], Regions())
        midpoint = 1.5e7
        got = f["bO"][0](midpoint)
        interpolated_b = np.interp(midpoint, TABLE[:, 0], 1.0 / TABLE[:, 1])
        inverted = 1.0 / np.interp(midpoint, TABLE[:, 0], TABLE[:, 1])
        assert np.isclose(got, interpolated_b), got
        assert not np.isclose(got, inverted, rtol=1e-6), (got, inverted)
        print("ok test_b_interpolates_rather_than_inverting_an_interpolated_fvf")

    def test_one_interpolant_per_pvt_region() -> None:
        second = TABLE.copy()
        second[:, 1] *= 2.0
        f = assign_pvdo({}, [TABLE, second], Regions(pvt=2))
        assert len(f["bO"]) == 2
        assert np.isclose(f["bO"][1](2.0e7), 1.0 / (1.15 * 2.0))
        print("ok test_one_interpolant_per_pvt_region")

    def test_calling_it_like_a_live_oil_property_is_rejected() -> None:
        """Dead oil takes pressure alone; live oil takes (p, rs, flag).

        The captured table is a keyword-only argument precisely so this
        mistake raises. Bound positionally it would silently interpolate
        against `rs` and `flag`, which returns NaN rather than an error.
        """
        f = assign_pvdo({}, [TABLE], Regions())
        try:
            f["bO"][0](np.array([2.0e7]), np.array([50.0]), np.array([True]))
        except TypeError:
            print("ok test_calling_it_like_a_live_oil_property_is_rejected")
            return
        raise AssertionError("a dead-oil bO must not accept three arguments")

    def test_accepts_an_array_of_pressures() -> None:
        f = assign_pvdo({}, [TABLE], Regions())
        out = f["bO"][0](np.array([1.0e7, 3.0e7]))
        assert np.allclose(out, [1 / 1.20, 1 / 1.10]), out
        print("ok test_accepts_an_array_of_pressures")

    test_b_is_the_reciprocal_of_the_tabulated_fvf()
    test_viscosity_is_read_straight_from_the_table()
    test_b_interpolates_rather_than_inverting_an_interpolated_fvf()
    test_one_interpolant_per_pvt_region()
    test_calling_it_like_a_live_oil_property_is_rejected()
    test_accepts_an_array_of_pressures()
    print("\nAll tests passed!")
