"""Surfactant solution viscosity from SURFVISC (MRST ``props/assignSURFVISC.m``).

SURFVISC gives the water viscosity of a surfactant solution against
concentration. Note this is a viscosity, not a multiplier - contrast
PLYVISC, which is a factor. Treating one as the other is off by the
viscosity of water.
"""

from typing import Any, Callable, Dict, List

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab


def assign_surfvisc(f: Dict[str, Any], surfvisc, reg) -> Dict[str, Any]:
    """Store ``muWSft`` as one interpolant per PVT region."""
    curves: List[Callable] = []
    for index in range(reg.pvt):
        table = extend_tab(np.asarray(surfvisc[index], dtype=float))
        curves.append(
            lambda x, *, a=table[:, 0], b=table[:, 1]: reg.interp(a, b, x))
    f["muWSft"] = curves
    return f


assignSURFVISC = assign_surfvisc


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    TABLE = np.array([[0.0, 5.0e-4],
                      [1.0, 8.0e-4],
                      [5.0, 2.0e-3]])

    def test_the_curve_is_read_from_the_table() -> None:
        f = assign_surfvisc({}, [TABLE], Regions())
        assert np.isclose(f["muWSft"][0](1.0), 8.0e-4)
        print("ok test_the_curve_is_read_from_the_table")

    def test_it_interpolates_between_rows() -> None:
        f = assign_surfvisc({}, [TABLE], Regions())
        midpoint = (TABLE[0, 0] + TABLE[1, 0]) / 2
        expected = (TABLE[0, 1] + TABLE[1, 1]) / 2
        assert np.isclose(f["muWSft"][0](midpoint), expected)
        print("ok test_it_interpolates_between_rows")

    def test_it_is_flat_outside_the_table() -> None:
        """A Newton step can propose a value outside the tabulated range."""
        f = assign_surfvisc({}, [TABLE], Regions())
        assert np.isclose(f["muWSft"][0](TABLE[0, 0] - 10.0), TABLE[0, 1])
        assert np.isclose(f["muWSft"][0](TABLE[-1, 0] + 10.0), TABLE[-1, 1])
        print("ok test_it_is_flat_outside_the_table")

    def test_it_accepts_an_array() -> None:
        f = assign_surfvisc({}, [TABLE], Regions())
        out = f["muWSft"][0](TABLE[:, 0])
        assert np.allclose(out, TABLE[:, 1]), out
        print("ok test_it_accepts_an_array")

    def test_one_curve_per_region() -> None:
        second = TABLE.copy()
        second[:, 1] *= 2.0
        f = assign_surfvisc({}, [TABLE, second], Regions(pvt=2))
        assert len(f["muWSft"]) == 2
        assert np.isclose(f["muWSft"][1](1.0), 8.0e-4 * 2.0)
        print("ok test_one_curve_per_region")

    test_the_curve_is_read_from_the_table()
    test_it_interpolates_between_rows()
    test_it_is_flat_outside_the_table()
    test_it_accepts_an_array()
    test_one_curve_per_region()
    print("\nAll tests passed!")
