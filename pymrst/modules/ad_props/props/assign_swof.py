"""Water-oil saturation functions from SWOF (MRST ``props/assignSWOF.m``).

SWOF tabulates, against water saturation: water relative permeability, oil
relative permeability in the water-oil system, and the oil-water capillary
pressure.

Two things here are less obvious than they look.

**krOW is a function of oil saturation, not water saturation.** The deck
tabulates it against Sw, but every caller asks for it at So. The table is
therefore reversed and re-indexed on So = 1 - Sw, which also restores an
increasing abscissa - an interpolator needs one, and reversing the rows is
what provides it.

**Capillary pressure is only attached when the deck states one.** A table of
zeros is not the same as no table: `pcOW` present-and-zero makes downstream
code take its capillary-pressure path for no reason, so MRST omits the field
entirely when every entry is zero, and so does this.

The saturation end points (`krPts`) are computed before the table is extended,
because they describe the deck's own table, not the padded one.
"""

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab


def _points(swof: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """End points of the water and oil-in-water curves.

    Each is (connate, critical, maximum saturation, maximum relperm).
    """
    water = np.zeros(4)
    water[0] = swof[0, 0]                                  # connate water
    immobile = np.flatnonzero(swof[:, 1] == 0)
    water[1] = swof[immobile[-1], 0] if immobile.size else swof[0, 0]
    water[2] = swof[-1, 0]
    water[3] = swof[-1, 1]                                 # maximum krW

    oil = np.zeros(4)
    oil[2] = 1.0
    dry = np.flatnonzero(swof[:, 2] <= np.finfo(float).eps)
    oil[1] = 1.0 - swof[dry[0], 0] if dry.size else 1.0 - swof[-1, 0]
    oil[3] = swof[0, 2]                                    # maximum krOW
    return water, oil


def assign_swof(f: Dict[str, Any], swof, reg) -> Dict[str, Any]:
    """Store ``krW``, ``krOW``, ``krPts`` and - if stated - ``pcOW``."""
    kr_w: List[Callable] = []
    kr_ow: List[Callable] = []
    pc_ow: List[Callable] = []
    pts_w = np.zeros((reg.sat, 4))
    pts_ow = np.zeros((reg.sat, 4))
    has_pc = False

    for index in range(reg.sat):
        table = np.asarray(swof[index], dtype=float)
        pts_w[index], pts_ow[index] = _points(table)

        table = extend_tab(table)
        sw = table[:, 0]
        pc = table[:, 3]
        has_pc = has_pc or bool(np.any(pc != 0))

        # krOW is asked for at So, so reverse the table onto So = 1 - Sw.
        so = 1.0 - sw[::-1]
        krow = table[::-1, 2]

        kr_w.append(lambda s, *, x=sw, y=table[:, 1]: reg.interp(x, y, s))
        kr_ow.append(lambda s, *, x=so, y=krow: reg.interp(x, y, s))
        pc_ow.append(lambda s, *, x=sw, y=pc: reg.interp(x, y, s))

    f["krW"], f["krOW"] = kr_w, kr_ow
    f.setdefault("krPts", {})
    f["krPts"]["w"], f["krPts"]["ow"] = pts_w, pts_ow
    if has_pc:
        f["pcOW"] = pc_ow
    return f


assignSWOF = assign_swof


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    #        Sw    krW   krOW   Pcow
    TABLE = np.array([[0.12, 0.00, 1.00, 0.0],
                      [0.30, 0.05, 0.60, 0.0],
                      [0.60, 0.30, 0.15, 0.0],
                      [0.90, 1.00, 0.00, 0.0]])

    def test_krw_is_read_at_water_saturation() -> None:
        f = assign_swof({}, [TABLE], Regions())
        assert np.isclose(f["krW"][0](0.30), 0.05)
        assert np.isclose(f["krW"][0](0.90), 1.00)
        print("ok test_krw_is_read_at_water_saturation")

    def test_krow_is_read_at_oil_saturation() -> None:
        """The reversal: krOW(So), not krOW(Sw)."""
        f = assign_swof({}, [TABLE], Regions())
        assert np.isclose(f["krOW"][0](1.0 - 0.30), 0.60)
        assert np.isclose(f["krOW"][0](1.0 - 0.60), 0.15)
        print("ok test_krow_is_read_at_oil_saturation")

    def test_krow_is_not_accidentally_a_function_of_sw() -> None:
        f = assign_swof({}, [TABLE], Regions())
        assert not np.isclose(f["krOW"][0](0.30), 0.60), "would mean no reversal"
        print("ok test_krow_is_not_accidentally_a_function_of_sw")

    def test_relperms_stay_flat_outside_the_table() -> None:
        f = assign_swof({}, [TABLE], Regions())
        assert np.isclose(f["krW"][0](0.05), 0.0)
        assert np.isclose(f["krW"][0](1.0), 1.0)
        print("ok test_relperms_stay_flat_outside_the_table")

    def test_no_capillary_pressure_field_when_the_deck_states_none() -> None:
        f = assign_swof({}, [TABLE], Regions())
        assert "pcOW" not in f, "an all-zero column is not a Pc curve"
        print("ok test_no_capillary_pressure_field_when_the_deck_states_none")

    def test_capillary_pressure_is_attached_when_stated() -> None:
        table = TABLE.copy()
        table[:, 3] = [3.0e4, 1.0e4, 2.0e3, 0.0]
        f = assign_swof({}, [table], Regions())
        assert "pcOW" in f
        assert np.isclose(f["pcOW"][0](0.30), 1.0e4)
        print("ok test_capillary_pressure_is_attached_when_stated")

    def test_end_points_describe_the_deck_table() -> None:
        f = assign_swof({}, [TABLE], Regions())
        water = f["krPts"]["w"][0]
        assert np.isclose(water[0], 0.12), "connate water"
        assert np.isclose(water[1], 0.12), "last immobile water"
        assert np.isclose(water[2], 0.90), "maximum tabulated Sw"
        assert np.isclose(water[3], 1.00), "maximum krW"
        print("ok test_end_points_describe_the_deck_table")

    def test_oil_end_points_are_expressed_in_oil_saturation() -> None:
        f = assign_swof({}, [TABLE], Regions())
        oil = f["krPts"]["ow"][0]
        assert np.isclose(oil[2], 1.0)
        assert np.isclose(oil[1], 1.0 - 0.90), "residual oil, from krOW = 0"
        assert np.isclose(oil[3], 1.00), "maximum krOW"
        print("ok test_oil_end_points_are_expressed_in_oil_saturation")

    def test_end_points_are_unaffected_by_the_table_extension() -> None:
        """They describe the deck's table, so padding must not move them."""
        f = assign_swof({}, [TABLE], Regions())
        assert np.isclose(f["krPts"]["w"][0][0], TABLE[0, 0])
        assert np.isclose(f["krPts"]["w"][0][2], TABLE[-1, 0])
        print("ok test_end_points_are_unaffected_by_the_table_extension")

    def test_one_curve_per_saturation_region() -> None:
        second = TABLE.copy()
        second[:, 1] *= 0.5
        f = assign_swof({}, [TABLE, second], Regions(sat=2))
        assert len(f["krW"]) == 2
        assert np.isclose(f["krW"][1](0.30), 0.025)
        assert f["krPts"]["w"].shape == (2, 4)
        print("ok test_one_curve_per_saturation_region")

    test_krw_is_read_at_water_saturation()
    test_krow_is_read_at_oil_saturation()
    test_krow_is_not_accidentally_a_function_of_sw()
    test_relperms_stay_flat_outside_the_table()
    test_no_capillary_pressure_field_when_the_deck_states_none()
    test_capillary_pressure_is_attached_when_stated()
    test_end_points_describe_the_deck_table()
    test_oil_end_points_are_expressed_in_oil_saturation()
    test_end_points_are_unaffected_by_the_table_extension()
    test_one_curve_per_saturation_region()
    print("\nAll tests passed!")
