"""Dry-gas PVT from PVDG (MRST ``ad-props/props/assignPVDG.m``).

The same shape as PVDO, with one addition: the table is extended down to zero
pressure by repeating its first row.

That extension is not cosmetic. Gas tables typically start well above zero,
and a Newton step can propose a near-zero pressure in a gas cell long before
the iteration settles. Without the extra point the interpolant extrapolates
off the bottom of the table; how far wrong that goes depends on the table's
first two rows, and for a steep one it can reach zero or negative b, which
turns a recoverable bad step into a solver failure. Flat extension keeps the
property bounded whatever the table looks like, and lets the step be rejected
on its own terms.
"""

from typing import Any, Callable, Dict, List

import numpy as np


def assign_pvdg(f: Dict[str, Any], pvdg, reg) -> Dict[str, Any]:
    """Store ``bG`` and ``muG`` as one interpolant per PVT region."""
    b_functions: List[Callable] = []
    mu_functions: List[Callable] = []

    for index in range(reg.pvt):
        table = np.asarray(pvdg[index], dtype=float)
        # extend to p = 0 by repeating the first row
        pressure = np.concatenate([[0.0], table[:, 0]])
        b = 1.0 / table[:, 1]
        viscosity = table[:, 2]
        b = np.concatenate([b[:1], b])
        viscosity = np.concatenate([viscosity[:1], viscosity])

        b_functions.append(
            lambda p, *, x=pressure, y=b: reg.interp(x, y, p))
        mu_functions.append(
            lambda p, *, x=pressure, y=viscosity: reg.interp(x, y, p))

    f["bG"], f["muG"] = b_functions, mu_functions
    return f


assignPVDG = assign_pvdg


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    TABLE = np.array([[1.0e7, 5.0e-3, 1.4e-5],
                      [2.0e7, 3.0e-3, 1.7e-5],
                      [3.0e7, 2.5e-3, 2.0e-5]])

    def test_b_is_the_reciprocal_of_the_tabulated_fvf() -> None:
        f = assign_pvdg({}, [TABLE], Regions())
        assert np.isclose(f["bG"][0](2.0e7), 1.0 / 3.0e-3)
        print("ok test_b_is_the_reciprocal_of_the_tabulated_fvf")

    def test_the_table_is_flat_below_its_first_pressure() -> None:
        f = assign_pvdg({}, [TABLE], Regions())
        first = 1.0 / TABLE[0, 1]
        assert np.isclose(f["bG"][0](0.0), first)
        assert np.isclose(f["bG"][0](5.0e6), first), "flat, not extrapolated"
        print("ok test_the_table_is_flat_below_its_first_pressure")

    def test_b_stays_positive_at_zero_pressure() -> None:
        f = assign_pvdg({}, [TABLE], Regions())
        assert f["bG"][0](0.0) > 0.0
        print("ok test_b_stays_positive_at_zero_pressure")

    def test_a_steep_table_would_extrapolate_to_a_negative_b() -> None:
        """What the extension protects against, on a table where it bites.

        Whether linear extrapolation reaches zero depends on the first two
        rows, so this uses a table steep enough to show it rather than
        asserting it of every table.
        """
        steep = np.array([[2.0e7, 5.0e-3, 1.4e-5],
                          [2.5e7, 1.0e-3, 1.7e-5]])
        slope = ((1 / steep[1, 1] - 1 / steep[0, 1])
                 / (steep[1, 0] - steep[0, 0]))
        extrapolated = 1 / steep[0, 1] - slope * steep[0, 0]
        assert extrapolated < 0.0, extrapolated
        f = assign_pvdg({}, [steep], Regions())
        assert np.isclose(f["bG"][0](0.0), 1 / steep[0, 1]), "flat instead"
        print("ok test_a_steep_table_would_extrapolate_to_a_negative_b")

    def test_viscosity_is_extended_the_same_way() -> None:
        f = assign_pvdg({}, [TABLE], Regions())
        assert np.isclose(f["muG"][0](0.0), TABLE[0, 2])
        print("ok test_viscosity_is_extended_the_same_way")

    def test_values_inside_the_table_are_untouched() -> None:
        f = assign_pvdg({}, [TABLE], Regions())
        for row in TABLE:
            assert np.isclose(f["bG"][0](row[0]), 1.0 / row[1])
            assert np.isclose(f["muG"][0](row[0]), row[2])
        print("ok test_values_inside_the_table_are_untouched")

    def test_one_interpolant_per_pvt_region() -> None:
        f = assign_pvdg({}, [TABLE, TABLE * 1.1], Regions(pvt=2))
        assert len(f["bG"]) == 2 and len(f["muG"]) == 2
        print("ok test_one_interpolant_per_pvt_region")

    test_b_is_the_reciprocal_of_the_tabulated_fvf()
    test_the_table_is_flat_below_its_first_pressure()
    test_b_stays_positive_at_zero_pressure()
    test_a_steep_table_would_extrapolate_to_a_negative_b()
    test_viscosity_is_extended_the_same_way()
    test_values_inside_the_table_are_untouched()
    test_one_interpolant_per_pvt_region()
    print("\nAll tests passed!")
