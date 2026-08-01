"""Water saturation function from SWFN (MRST ``props/assignSWFN.m``).

SWFN is the three-column family's water table: Sw, krW, Pcow. Unlike SWOF it
says nothing about oil - that comes from SOF2 or SOF3 - so a deck using this
family needs both keywords, and this file assigns only the water side.

When the deck has a surfactant region, three further fields are attached. The
surfactant relative permeability reads the same table through the *surfactant*
region map rather than the saturation one, because a surfactant region can
partition the field differently; and the residual saturations are recorded for
both partitions. `sOres` is derived as 1 - Sw_max, the largest tabulated water
saturation being where oil stops flowing.
"""

from typing import Any, Callable, Dict, List

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab
from pymrst.modules.ad_props.utils.get_reg_map import get_reg_map
from pymrst.modules.ad_props.utils.interp_reg import interp_reg


def _points(swfn: np.ndarray) -> np.ndarray:
    """(connate, last immobile, maximum Sw, maximum krW)."""
    pts = np.zeros(4)
    pts[0] = swfn[0, 0]
    immobile = np.flatnonzero(swfn[:, 1] == 0)
    pts[1] = swfn[immobile[-1], 0] if immobile.size else swfn[0, 0]
    pts[2] = swfn[-1, 0]
    pts[3] = swfn[-1, 1]
    return pts


def assign_swfn(f: Dict[str, Any], swfn, reg) -> Dict[str, Any]:
    """Store ``krW``, ``krPts.w`` and - if stated - ``pcOW``."""
    kr_w: List[Callable] = []
    pc_ow: List[Callable] = []
    pts_w = np.zeros((reg.sat, 4))
    has_pc = False

    for index in range(reg.sat):
        table = np.asarray(swfn[index], dtype=float)
        pts_w[index] = _points(table)

        table = extend_tab(table)
        sw, kr, pc = table[:, 0], table[:, 1], table[:, 2]
        has_pc = has_pc or bool(np.any(pc != 0))
        kr_w.append(lambda s, *, x=sw, y=kr: reg.interp(x, y, s))
        pc_ow.append(lambda s, *, x=sw, y=pc: reg.interp(x, y, s))

    f["krW"] = kr_w
    f.setdefault("krPts", {})["w"] = pts_w
    if has_pc:
        f["pcOW"] = pc_ow

    if reg.SURFNUM is not None:
        tables = [extend_tab(np.asarray(t, dtype=float)[:, :2]) for t in swfn]

        def kr_w_sft(sw, cell_inx=None, _t=tables):
            index = get_reg_map(sw, reg.SURFNUM, reg.SURFINX,
                                cell_inx=cell_inx)
            return interp_reg(_t, sw, index)

        f["krWSft"] = kr_w_sft
        f["sWconSft"] = pts_w[:, 0][reg.SURFNUM]
        # oil stops flowing at the largest tabulated water saturation
        s_ores = np.array([np.asarray(t, dtype=float)[-1, 0] for t in swfn])
        f["sOres"] = 1.0 - s_ores[reg.SATNUM]
        f["sOresSft"] = 1.0 - s_ores[reg.SURFNUM]
    return f


assignSWFN = assign_swfn


if __name__ == "__main__":
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    #        Sw    krW   Pcow
    TABLE = np.array([[0.15, 0.00, 3.0e4],
                      [0.40, 0.10, 8.0e3],
                      [0.70, 0.45, 1.0e3],
                      [0.85, 1.00, 0.00]])

    def test_krw_is_read_at_water_saturation() -> None:
        f = assign_swfn({}, [TABLE], Regions())
        assert np.isclose(f["krW"][0](0.40), 0.10)
        print("ok test_krw_is_read_at_water_saturation")

    def test_capillary_pressure_is_attached_when_stated() -> None:
        f = assign_swfn({}, [TABLE], Regions())
        assert "pcOW" in f and np.isclose(f["pcOW"][0](0.70), 1.0e3)
        print("ok test_capillary_pressure_is_attached_when_stated")

    def test_no_capillary_pressure_field_for_an_all_zero_column() -> None:
        table = TABLE.copy()
        table[:, 2] = 0.0
        assert "pcOW" not in assign_swfn({}, [table], Regions())
        print("ok test_no_capillary_pressure_field_for_an_all_zero_column")

    def test_end_points_describe_the_deck_table() -> None:
        f = assign_swfn({}, [TABLE], Regions())
        pts = f["krPts"]["w"][0]
        assert np.allclose(pts, [0.15, 0.15, 0.85, 1.00]), pts
        print("ok test_end_points_describe_the_deck_table")

    def test_relperm_is_flat_outside_the_table() -> None:
        f = assign_swfn({}, [TABLE], Regions())
        assert np.isclose(f["krW"][0](0.05), 0.0)
        assert np.isclose(f["krW"][0](1.00), 1.0)
        print("ok test_relperm_is_flat_outside_the_table")

    def test_no_surfactant_fields_without_a_surfactant_region() -> None:
        f = assign_swfn({}, [TABLE], Regions())
        assert "krWSft" not in f and "sOres" not in f
        print("ok test_no_surfactant_fields_without_a_surfactant_region")

    def test_surfactant_fields_use_the_surfactant_partition() -> None:
        """krWSft reads the same table through SURFNUM, not SATNUM."""
        second = TABLE.copy()
        second[:, 1] *= 0.5
        reg = Regions(sat=2,
                      SATNUM=np.array([0, 0, 1, 1]),
                      SATINX=[np.array([0, 1]), np.array([2, 3])],
                      SURFNUM=np.array([0, 1, 0, 1]),
                      SURFINX=[np.array([0, 2]), np.array([1, 3])])
        f = assign_swfn({}, [TABLE, second], reg)
        got = f["krWSft"](np.full(4, 0.40))
        assert np.allclose(got, [0.10, 0.05, 0.10, 0.05]), got
        print("ok test_surfactant_fields_use_the_surfactant_partition")

    def test_residual_oil_is_one_minus_maximum_water() -> None:
        reg = Regions(SATNUM=np.array([0, 0]), SATINX=[np.array([0, 1])],
                      SURFNUM=np.array([0, 0]), SURFINX=[np.array([0, 1])])
        f = assign_swfn({}, [TABLE], reg)
        assert np.allclose(f["sOres"], 1.0 - 0.85)
        assert np.allclose(f["sOresSft"], 1.0 - 0.85)
        assert np.allclose(f["sWconSft"], 0.15)
        print("ok test_residual_oil_is_one_minus_maximum_water")

    def test_one_curve_per_saturation_region() -> None:
        f = assign_swfn({}, [TABLE, TABLE], Regions(sat=2))
        assert len(f["krW"]) == 2 and f["krPts"]["w"].shape == (2, 4)
        print("ok test_one_curve_per_saturation_region")

    test_krw_is_read_at_water_saturation()
    test_capillary_pressure_is_attached_when_stated()
    test_no_capillary_pressure_field_for_an_all_zero_column()
    test_end_points_describe_the_deck_table()
    test_relperm_is_flat_outside_the_table()
    test_no_surfactant_fields_without_a_surfactant_region()
    test_surfactant_fields_use_the_surfactant_partition()
    test_residual_oil_is_one_minus_maximum_water()
    test_one_curve_per_saturation_region()
    print("\nAll tests passed!")
