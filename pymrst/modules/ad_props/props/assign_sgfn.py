"""Gas saturation function from SGFN (MRST ``props/assignSGFN.m``).

SGFN is the three-column family's gas table: Sg, krG, Pcog. Like SWFN it says
nothing about oil, which comes from SOF2 or SOF3.

It reads the connate water saturation from SWFN's end points, as SGOF does
from SWOF's - but note what it is used for here. SGFN's own end points do not
involve Swcon at all; MRST reads it and then only passes it to `getPoints`,
which ignores it. The dependency is therefore vestigial in this keyword, and
this port keeps the check rather than the use: assigning SGFN before the water
table still signals a caller that has the order wrong, which matters for the
oil tables that follow.
"""

from typing import Any, Callable, Dict, List

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab


def _points(sgfn: np.ndarray) -> np.ndarray:
    """(connate, last immobile, maximum Sg, maximum krG)."""
    pts = np.zeros(4)
    pts[0] = sgfn[0, 0]
    immobile = np.flatnonzero(sgfn[:, 1] == 0)
    pts[1] = sgfn[immobile[-1], 0] if immobile.size else sgfn[0, 0]
    pts[2] = sgfn[-1, 0]
    pts[3] = sgfn[-1, 1]
    return pts


def assign_sgfn(f: Dict[str, Any], sgfn, reg) -> Dict[str, Any]:
    """Store ``krG``, ``krPts.g`` and - if stated - ``pcOG``."""
    if "krPts" not in f or "w" not in f["krPts"]:
        raise ValueError(
            "SGFN expects the water table to be assigned first; assign SWFN "
            "(or SWOF) before it")

    kr_g: List[Callable] = []
    pc_og: List[Callable] = []
    pts_g = np.zeros((reg.sat, 4))
    has_pc = False

    for index in range(reg.sat):
        table = np.asarray(sgfn[index], dtype=float)
        pts_g[index] = _points(table)

        table = extend_tab(table)
        sg, kr, pc = table[:, 0], table[:, 1], table[:, 2]
        has_pc = has_pc or bool(np.any(pc != 0))
        kr_g.append(lambda s, *, x=sg, y=kr: reg.interp(x, y, s))
        pc_og.append(lambda s, *, x=sg, y=pc: reg.interp(x, y, s))

    f["krG"] = kr_g
    f.setdefault("krPts", {})["g"] = pts_g
    if has_pc:
        f["pcOG"] = pc_og
    return f


assignSGFN = assign_sgfn


if __name__ == "__main__":
    from pymrst.modules.ad_props.props.assign_swfn import assign_swfn
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    SWFN = np.array([[0.15, 0.00, 3.0e4],
                     [0.40, 0.10, 8.0e3],
                     [0.85, 1.00, 0.00]])
    #        Sg    krG   Pcog
    TABLE = np.array([[0.00, 0.00, 0.0],
                      [0.10, 0.00, 1.0e3],
                      [0.40, 0.25, 6.0e3],
                      [0.80, 0.90, 2.0e4]])

    def _base(reg=None):
        reg = reg or Regions()
        return assign_swfn({}, [SWFN] * reg.sat, reg), reg

    def test_krg_is_read_at_gas_saturation() -> None:
        f, reg = _base()
        f = assign_sgfn(f, [TABLE], reg)
        assert np.isclose(f["krG"][0](0.40), 0.25)
        print("ok test_krg_is_read_at_gas_saturation")

    def test_last_immobile_gas_is_the_last_zero_not_the_first() -> None:
        """Two rows have krG = 0; the end point is the upper one."""
        f, reg = _base()
        f = assign_sgfn(f, [TABLE], reg)
        assert np.isclose(f["krPts"]["g"][0][1], 0.10)
        print("ok test_last_immobile_gas_is_the_last_zero_not_the_first")

    def test_end_points_describe_the_deck_table() -> None:
        f, reg = _base()
        f = assign_sgfn(f, [TABLE], reg)
        assert np.allclose(f["krPts"]["g"][0], [0.00, 0.10, 0.80, 0.90])
        print("ok test_end_points_describe_the_deck_table")

    def test_capillary_pressure_is_attached_when_stated() -> None:
        f, reg = _base()
        f = assign_sgfn(f, [TABLE], reg)
        assert "pcOG" in f and np.isclose(f["pcOG"][0](0.40), 6.0e3)
        print("ok test_capillary_pressure_is_attached_when_stated")

    def test_no_capillary_pressure_field_for_an_all_zero_column() -> None:
        table = TABLE.copy()
        table[:, 2] = 0.0
        f, reg = _base()
        assert "pcOG" not in assign_sgfn(f, [table], reg)
        print("ok test_no_capillary_pressure_field_for_an_all_zero_column")

    def test_assigning_before_the_water_table_is_rejected() -> None:
        try:
            assign_sgfn({}, [TABLE], Regions())
        except ValueError as error:
            assert "SWFN" in str(error)
            print("ok test_assigning_before_the_water_table_is_rejected")
            return
        raise AssertionError("SGFN without a water table must be rejected")

    def test_the_water_points_survive() -> None:
        f, reg = _base()
        f = assign_sgfn(f, [TABLE], reg)
        assert set(f["krPts"]) == {"w", "g"}, set(f["krPts"])
        print("ok test_the_water_points_survive")

    def test_one_curve_per_saturation_region() -> None:
        f, reg = _base(Regions(sat=2))
        f = assign_sgfn(f, [TABLE, TABLE], reg)
        assert len(f["krG"]) == 2 and f["krPts"]["g"].shape == (2, 4)
        print("ok test_one_curve_per_saturation_region")

    test_krg_is_read_at_gas_saturation()
    test_last_immobile_gas_is_the_last_zero_not_the_first()
    test_end_points_describe_the_deck_table()
    test_capillary_pressure_is_attached_when_stated()
    test_no_capillary_pressure_field_for_an_all_zero_column()
    test_assigning_before_the_water_table_is_rejected()
    test_the_water_points_survive()
    test_one_curve_per_saturation_region()
    print("\nAll tests passed!")
