"""Polymer adsorption from PLYADS (MRST ``props/assignPLYADS.m``).

PLYADS gives adsorbed polymer against concentration in solution.
Adsorption removes polymer from the flowing phase and is what makes a
polymer flood consume more polymer than the swept volume suggests.

It partitions on the saturation regions, not the PVT ones: adsorption is
a rock property. MRST extends the tables twice - once on the whole list
and once per table inside the loop - which changes nothing, because both
extensions are flat, so this extends once.
"""

from typing import Any, Callable, Dict, List

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab


def assign_plyads(f: Dict[str, Any], plyads, reg) -> Dict[str, Any]:
    """Store ``ads`` as one interpolant per saturation region."""
    curves: List[Callable] = []
    for index in range(reg.sat):
        table = extend_tab(np.asarray(plyads[index], dtype=float))
        curves.append(
            lambda x, *, a=table[:, 0], b=table[:, 1]: reg.interp(a, b, x))
    f["ads"] = curves
    return f


assignPLYADS = assign_plyads


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    TABLE = np.array([[0.0, 0.0],
                      [1.0, 0.002],
                      [3.0, 0.003]])

    def test_the_curve_is_read_from_the_table() -> None:
        f = assign_plyads({}, [TABLE], Regions())
        assert np.isclose(f["ads"][0](1.0), 0.002)
        print("ok test_the_curve_is_read_from_the_table")

    def test_it_interpolates_between_rows() -> None:
        f = assign_plyads({}, [TABLE], Regions())
        midpoint = (TABLE[0, 0] + TABLE[1, 0]) / 2
        expected = (TABLE[0, 1] + TABLE[1, 1]) / 2
        assert np.isclose(f["ads"][0](midpoint), expected)
        print("ok test_it_interpolates_between_rows")

    def test_it_is_flat_outside_the_table() -> None:
        """A Newton step can propose a value outside the tabulated range."""
        f = assign_plyads({}, [TABLE], Regions())
        assert np.isclose(f["ads"][0](TABLE[0, 0] - 10.0), TABLE[0, 1])
        assert np.isclose(f["ads"][0](TABLE[-1, 0] + 10.0), TABLE[-1, 1])
        print("ok test_it_is_flat_outside_the_table")

    def test_it_accepts_an_array() -> None:
        f = assign_plyads({}, [TABLE], Regions())
        out = f["ads"][0](TABLE[:, 0])
        assert np.allclose(out, TABLE[:, 1]), out
        print("ok test_it_accepts_an_array")

    def test_one_curve_per_region() -> None:
        second = TABLE.copy()
        second[:, 1] *= 2.0
        f = assign_plyads({}, [TABLE, second], Regions(sat=2))
        assert len(f["ads"]) == 2
        assert np.isclose(f["ads"][1](1.0), 0.002 * 2.0)
        print("ok test_one_curve_per_region")

    test_the_curve_is_read_from_the_table()
    test_it_interpolates_between_rows()
    test_it_is_flat_outside_the_table()
    test_it_accepts_an_array()
    test_one_curve_per_region()
    print("\nAll tests passed!")
