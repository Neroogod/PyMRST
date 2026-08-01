"""Surfactant adsorption from SURFADS (MRST ``props/assignSURFADS.m``).

SURFADS gives adsorbed surfactant against concentration in solution,
the surfactant counterpart of PLYADS. Adsorption on the rock is the
main loss mechanism in a surfactant flood.
"""

from typing import Any, Callable, Dict, List

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab


def assign_surfads(f: Dict[str, Any], surfads, reg) -> Dict[str, Any]:
    """Store ``surfads`` as one interpolant per saturation region."""
    curves: List[Callable] = []
    for index in range(reg.sat):
        table = extend_tab(np.asarray(surfads[index], dtype=float))
        curves.append(
            lambda x, *, a=table[:, 0], b=table[:, 1]: reg.interp(a, b, x))
    f["surfads"] = curves
    return f


assignSURFADS = assign_surfads


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    TABLE = np.array([[0.0, 0.0],
                      [1.0, 0.0005],
                      [5.0, 0.001]])

    def test_the_curve_is_read_from_the_table() -> None:
        f = assign_surfads({}, [TABLE], Regions())
        assert np.isclose(f["surfads"][0](1.0), 0.0005)
        print("ok test_the_curve_is_read_from_the_table")

    def test_it_interpolates_between_rows() -> None:
        f = assign_surfads({}, [TABLE], Regions())
        midpoint = (TABLE[0, 0] + TABLE[1, 0]) / 2
        expected = (TABLE[0, 1] + TABLE[1, 1]) / 2
        assert np.isclose(f["surfads"][0](midpoint), expected)
        print("ok test_it_interpolates_between_rows")

    def test_it_is_flat_outside_the_table() -> None:
        """A Newton step can propose a value outside the tabulated range."""
        f = assign_surfads({}, [TABLE], Regions())
        assert np.isclose(f["surfads"][0](TABLE[0, 0] - 10.0), TABLE[0, 1])
        assert np.isclose(f["surfads"][0](TABLE[-1, 0] + 10.0), TABLE[-1, 1])
        print("ok test_it_is_flat_outside_the_table")

    def test_it_accepts_an_array() -> None:
        f = assign_surfads({}, [TABLE], Regions())
        out = f["surfads"][0](TABLE[:, 0])
        assert np.allclose(out, TABLE[:, 1]), out
        print("ok test_it_accepts_an_array")

    def test_one_curve_per_region() -> None:
        second = TABLE.copy()
        second[:, 1] *= 2.0
        f = assign_surfads({}, [TABLE, second], Regions(sat=2))
        assert len(f["surfads"]) == 2
        assert np.isclose(f["surfads"][1](1.0), 0.0005 * 2.0)
        print("ok test_one_curve_per_region")

    test_the_curve_is_read_from_the_table()
    test_it_interpolates_between_rows()
    test_it_is_flat_outside_the_table()
    test_it_accepts_an_array()
    test_one_curve_per_region()
    print("\nAll tests passed!")
