"""Oil saturation functions from SOF3 (MRST ``props/assignSOF3.m``).

SOF3 completes the three-column family: one table of oil saturation against
*both* oil relative permeabilities - krOW in the water-oil system and krOG in
the gas-oil system. SWFN and SGFN supply the water and gas sides.

The abscissa is already So, so unlike SWOF and SGOF there is no reversal and
no connate-water shift here: the deck states the curves the way they are asked
for.

The end points are read differently from the other keywords, and the
difference is easy to miss. Where SWFN takes the *last* saturation with zero
relative permeability, SOF3 takes the **first** - the residual oil saturation,
below which oil does not flow. Taking the last would return the top of the
table for any curve that is zero only at its start.
"""

from typing import Any, Callable, Dict, List

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab


def _points(so: np.ndarray, kr: np.ndarray) -> np.ndarray:
    """(unused, residual oil, 1, maximum krO)."""
    pts = np.zeros(4)
    zero = np.flatnonzero(kr == 0)
    pts[1] = so[zero[0]] if zero.size else so[0]
    pts[2] = 1.0
    pts[3] = kr[-1]
    return pts


def assign_sof3(f: Dict[str, Any], sof3, reg) -> Dict[str, Any]:
    """Store ``krOW``, ``krOG`` and their end points."""
    kr_ow: List[Callable] = []
    kr_og: List[Callable] = []
    pts_ow = np.zeros((reg.sat, 4))
    pts_og = np.zeros((reg.sat, 4))

    for index in range(reg.sat):
        table = np.asarray(sof3[index], dtype=float)
        pts_ow[index] = _points(table[:, 0], table[:, 1])
        pts_og[index] = _points(table[:, 0], table[:, 2])

        table = extend_tab(table)
        so, krow, krog = table[:, 0], table[:, 1], table[:, 2]
        kr_ow.append(lambda s, *, x=so, y=krow: reg.interp(x, y, s))
        kr_og.append(lambda s, *, x=so, y=krog: reg.interp(x, y, s))

    f["krOW"], f["krOG"] = kr_ow, kr_og
    f.setdefault("krPts", {})
    f["krPts"]["ow"], f["krPts"]["og"] = pts_ow, pts_og
    return f


assignSOF3 = assign_sof3


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    #        So    krOW   krOG
    TABLE = np.array([[0.00, 0.00, 0.00],
                      [0.20, 0.00, 0.00],
                      [0.50, 0.30, 0.25],
                      [0.85, 1.00, 0.90]])

    def test_both_curves_are_read_at_oil_saturation() -> None:
        f = assign_sof3({}, [TABLE], Regions())
        assert np.isclose(f["krOW"][0](0.50), 0.30)
        assert np.isclose(f["krOG"][0](0.50), 0.25)
        print("ok test_both_curves_are_read_at_oil_saturation")

    def test_residual_oil_is_the_first_zero_not_the_last() -> None:
        """SOF3 reverses the rule SWFN uses; this is the distinguishing case."""
        f = assign_sof3({}, [TABLE], Regions())
        assert np.isclose(f["krPts"]["ow"][0][1], 0.00), "first zero"
        assert not np.isclose(f["krPts"]["ow"][0][1], 0.20), "not the last"
        print("ok test_residual_oil_is_the_first_zero_not_the_last")

    def test_end_points_report_maximum_relperm_and_unit_saturation() -> None:
        f = assign_sof3({}, [TABLE], Regions())
        assert np.isclose(f["krPts"]["ow"][0][2], 1.0)
        assert np.isclose(f["krPts"]["ow"][0][3], 1.00)
        assert np.isclose(f["krPts"]["og"][0][3], 0.90)
        print("ok test_end_points_report_maximum_relperm_and_unit_saturation")

    def test_the_two_curves_have_separate_end_points() -> None:
        f = assign_sof3({}, [TABLE], Regions())
        assert not np.allclose(f["krPts"]["ow"][0], f["krPts"]["og"][0])
        print("ok test_the_two_curves_have_separate_end_points")

    def test_curves_are_flat_outside_the_table() -> None:
        f = assign_sof3({}, [TABLE], Regions())
        assert np.isclose(f["krOW"][0](-0.1), 0.0)
        assert np.isclose(f["krOW"][0](1.0), 1.00)
        print("ok test_curves_are_flat_outside_the_table")

    def test_no_connate_water_shift_is_applied() -> None:
        """SOF3's abscissa is already So, so it needs nothing from SWFN."""
        f = assign_sof3({}, [TABLE], Regions())
        assert np.isclose(f["krOG"][0](0.85), 0.90)
        print("ok test_no_connate_water_shift_is_applied")

    def test_one_pair_of_curves_per_saturation_region() -> None:
        second = TABLE.copy()
        second[:, 1] *= 0.5
        f = assign_sof3({}, [TABLE, second], Regions(sat=2))
        assert len(f["krOW"]) == 2
        assert np.isclose(f["krOW"][1](0.50), 0.15)
        assert f["krPts"]["ow"].shape == (2, 4)
        print("ok test_one_pair_of_curves_per_saturation_region")

    test_both_curves_are_read_at_oil_saturation()
    test_residual_oil_is_the_first_zero_not_the_last()
    test_end_points_report_maximum_relperm_and_unit_saturation()
    test_the_two_curves_have_separate_end_points()
    test_curves_are_flat_outside_the_table()
    test_no_connate_water_shift_is_applied()
    test_one_pair_of_curves_per_saturation_region()
    print("\nAll tests passed!")
