"""Polymer shear thinning from PLYSHEAR (MRST ``props/assignPLYSHEAR.m``).

PLYSHEAR gives a viscosity multiplier against water velocity. Polymer
solutions thin as they shear, so near a well - where velocity is highest
- the polymer is less viscous than the bulk table says.

The abscissa is a velocity, not a concentration or a saturation, so
unlike most tables here it has no natural bound.
"""

from typing import Any, Callable, Dict, List

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab


def assign_plyshear(f: Dict[str, Any], plyshear, reg) -> Dict[str, Any]:
    """Store ``plyshearMult`` as one interpolant per PVT region."""
    curves: List[Callable] = []
    for index in range(reg.pvt):
        table = extend_tab(np.asarray(plyshear[index], dtype=float))
        curves.append(
            lambda x, *, a=table[:, 0], b=table[:, 1]: reg.interp(a, b, x))
    f["plyshearMult"] = curves
    return f


assignPLYSHEAR = assign_plyshear


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    TABLE = np.array([[0.0, 1.0],
                      [1.0e-5, 0.9],
                      [1.0e-4, 0.5]])

    def test_the_curve_is_read_from_the_table() -> None:
        f = assign_plyshear({}, [TABLE], Regions())
        assert np.isclose(f["plyshearMult"][0](1.0e-5), 0.9)
        print("ok test_the_curve_is_read_from_the_table")

    def test_it_interpolates_between_rows() -> None:
        f = assign_plyshear({}, [TABLE], Regions())
        midpoint = (TABLE[0, 0] + TABLE[1, 0]) / 2
        expected = (TABLE[0, 1] + TABLE[1, 1]) / 2
        assert np.isclose(f["plyshearMult"][0](midpoint), expected)
        print("ok test_it_interpolates_between_rows")

    def test_it_is_flat_outside_the_table() -> None:
        """A Newton step can propose a value outside the tabulated range."""
        f = assign_plyshear({}, [TABLE], Regions())
        assert np.isclose(f["plyshearMult"][0](TABLE[0, 0] - 10.0), TABLE[0, 1])
        assert np.isclose(f["plyshearMult"][0](TABLE[-1, 0] + 10.0), TABLE[-1, 1])
        print("ok test_it_is_flat_outside_the_table")

    def test_it_accepts_an_array() -> None:
        f = assign_plyshear({}, [TABLE], Regions())
        out = f["plyshearMult"][0](TABLE[:, 0])
        assert np.allclose(out, TABLE[:, 1]), out
        print("ok test_it_accepts_an_array")

    def test_one_curve_per_region() -> None:
        second = TABLE.copy()
        second[:, 1] *= 2.0
        f = assign_plyshear({}, [TABLE, second], Regions(pvt=2))
        assert len(f["plyshearMult"]) == 2
        assert np.isclose(f["plyshearMult"][1](1.0e-5), 0.9 * 2.0)
        print("ok test_one_curve_per_region")

    test_the_curve_is_read_from_the_table()
    test_it_interpolates_between_rows()
    test_it_is_flat_outside_the_table()
    test_it_accepts_an_array()
    test_one_curve_per_region()
    print("\nAll tests passed!")
