"""Interfacial tension from SURFST (MRST ``props/assignSURFST.m``).

SURFST gives oil-water interfacial tension against surfactant
concentration. Lowering it is the whole point of a surfactant flood:
capillary forces trapping residual oil scale with it, so the curve
drives how much of that oil becomes mobile.
"""

from typing import Any, Callable, Dict, List

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab


def assign_surfst(f: Dict[str, Any], surfst, reg) -> Dict[str, Any]:
    """Store ``ift`` as one interpolant per PVT region."""
    curves: List[Callable] = []
    for index in range(reg.pvt):
        table = extend_tab(np.asarray(surfst[index], dtype=float))
        curves.append(
            lambda x, *, a=table[:, 0], b=table[:, 1]: reg.interp(a, b, x))
    f["ift"] = curves
    return f


assignSURFST = assign_surfst


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    TABLE = np.array([[0.0, 0.05],
                      [1.0, 0.001],
                      [5.0, 1.0e-5]])

    def test_the_curve_is_read_from_the_table() -> None:
        f = assign_surfst({}, [TABLE], Regions())
        assert np.isclose(f["ift"][0](1.0), 0.001)
        print("ok test_the_curve_is_read_from_the_table")

    def test_it_interpolates_between_rows() -> None:
        f = assign_surfst({}, [TABLE], Regions())
        midpoint = (TABLE[0, 0] + TABLE[1, 0]) / 2
        expected = (TABLE[0, 1] + TABLE[1, 1]) / 2
        assert np.isclose(f["ift"][0](midpoint), expected)
        print("ok test_it_interpolates_between_rows")

    def test_it_is_flat_outside_the_table() -> None:
        """A Newton step can propose a value outside the tabulated range."""
        f = assign_surfst({}, [TABLE], Regions())
        assert np.isclose(f["ift"][0](TABLE[0, 0] - 10.0), TABLE[0, 1])
        assert np.isclose(f["ift"][0](TABLE[-1, 0] + 10.0), TABLE[-1, 1])
        print("ok test_it_is_flat_outside_the_table")

    def test_it_accepts_an_array() -> None:
        f = assign_surfst({}, [TABLE], Regions())
        out = f["ift"][0](TABLE[:, 0])
        assert np.allclose(out, TABLE[:, 1]), out
        print("ok test_it_accepts_an_array")

    def test_one_curve_per_region() -> None:
        second = TABLE.copy()
        second[:, 1] *= 2.0
        f = assign_surfst({}, [TABLE, second], Regions(pvt=2))
        assert len(f["ift"]) == 2
        assert np.isclose(f["ift"][1](1.0), 0.001 * 2.0)
        print("ok test_one_curve_per_region")

    test_the_curve_is_read_from_the_table()
    test_it_interpolates_between_rows()
    test_it_is_flat_outside_the_table()
    test_it_accepts_an_array()
    test_one_curve_per_region()
    print("\nAll tests passed!")
