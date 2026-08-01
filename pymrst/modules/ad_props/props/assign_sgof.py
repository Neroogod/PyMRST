"""Gas-oil saturation functions from SGOF (MRST ``props/assignSGOF.m``).

SGOF is SWOF's gas-side counterpart, with one structural difference that makes
the order of assignment matter: **it needs the connate water saturation, which
comes from SWOF**.

The reason is physical. SGOF tabulates the oil relative permeability in the
gas-oil system against gas saturation, but in a three-phase cell the connate
water occupies pore space too, so the oil saturation at a given Sg is
1 - Sg - Swcon, not 1 - Sg. Assigning SGOF before SWOF therefore shifts the
whole krOG curve by the connate water saturation.

MRST warns and falls back to Swcon = 0 in that case rather than failing. This
port raises instead: a silent shift of a relative permeability curve is not
something a caller can be expected to notice, and the ordering is entirely
within the caller's control.
"""

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from pymrst.modules.ad_props.utils.extend_tab import extend_tab


def _points(sgof: np.ndarray, swcon: float) -> Tuple[np.ndarray, np.ndarray]:
    """End points of the gas and oil-in-gas curves."""
    gas = np.zeros(4)
    gas[0] = sgof[0, 0]                                    # connate gas
    immobile = np.flatnonzero(sgof[:, 1] == 0)
    gas[1] = sgof[immobile[-1], 0] if immobile.size else sgof[0, 0]
    gas[2] = sgof[-1, 0]
    gas[3] = sgof[-1, 1]                                   # maximum krG

    oil = np.zeros(4)
    oil[2] = 1.0
    dry = np.flatnonzero(sgof[:, 2] <= np.finfo(float).eps)
    oil[1] = (1.0 - sgof[dry[0], 0] - swcon if dry.size
              else 1.0 - sgof[-1, 0] - swcon)
    oil[3] = sgof[0, 2]                                    # maximum krOG
    return gas, oil


def assign_sgof(f: Dict[str, Any], sgof, reg) -> Dict[str, Any]:
    """Store ``krG``, ``krOG``, ``krPts`` and - if stated - ``pcOG``."""
    points = f.get("krPts", {}).get("w")
    if points is None:
        raise ValueError(
            "SGOF needs the connate water saturation from SWOF; assign SWOF "
            "first, or krOG is shifted by Swcon")

    kr_g: List[Callable] = []
    kr_og: List[Callable] = []
    pc_og: List[Callable] = []
    pts_g = np.zeros((reg.sat, 4))
    pts_og = np.zeros((reg.sat, 4))
    has_pc = False

    for index in range(reg.sat):
        swcon = float(points[index][0])
        table = np.asarray(sgof[index], dtype=float)
        pts_g[index], pts_og[index] = _points(table, swcon)

        table = extend_tab(table)
        sg = table[:, 0]
        pc = table[:, 3]
        has_pc = has_pc or bool(np.any(pc != 0))

        # krOG is asked for at So, and the water takes up Swcon of the pore.
        so = 1.0 - (sg[::-1] + swcon)
        krog = table[::-1, 2]

        kr_g.append(lambda s, *, x=sg, y=table[:, 1]: reg.interp(x, y, s))
        kr_og.append(lambda s, *, x=so, y=krog: reg.interp(x, y, s))
        pc_og.append(lambda s, *, x=sg, y=pc: reg.interp(x, y, s))

    f["krG"], f["krOG"] = kr_g, kr_og
    f.setdefault("krPts", {})
    f["krPts"]["g"], f["krPts"]["og"] = pts_g, pts_og
    if has_pc:
        f["pcOG"] = pc_og
    return f


assignSGOF = assign_sgof


if __name__ == "__main__":
    from pymrst.modules.ad_props.props.assign_swof import assign_swof
    from pymrst.modules.ad_props.utils.handle_regions import Regions

    SWOF = np.array([[0.12, 0.00, 1.00, 0.0],
                     [0.30, 0.05, 0.60, 0.0],
                     [0.60, 0.30, 0.15, 0.0],
                     [0.90, 1.00, 0.00, 0.0]])
    #         Sg    krG   krOG   Pcog
    SGOF = np.array([[0.00, 0.00, 1.00, 0.0],
                     [0.20, 0.05, 0.55, 0.0],
                     [0.50, 0.40, 0.10, 0.0],
                     [0.88, 1.00, 0.00, 0.0]])

    def _with_swof(reg=None):
        reg = reg or Regions()
        return assign_swof({}, [SWOF] * reg.sat, reg), reg

    def test_krg_is_read_at_gas_saturation() -> None:
        f, reg = _with_swof()
        f = assign_sgof(f, [SGOF], reg)
        assert np.isclose(f["krG"][0](0.20), 0.05)
        print("ok test_krg_is_read_at_gas_saturation")

    def test_krog_accounts_for_connate_water() -> None:
        """So = 1 - Sg - Swcon, so the curve sits Swcon lower than 1 - Sg."""
        f, reg = _with_swof()
        f = assign_sgof(f, [SGOF], reg)
        swcon = 0.12
        assert np.isclose(f["krOG"][0](1.0 - 0.20 - swcon), 0.55)
        print("ok test_krog_accounts_for_connate_water")

    def test_ignoring_connate_water_would_give_a_different_curve() -> None:
        f, reg = _with_swof()
        f = assign_sgof(f, [SGOF], reg)
        assert not np.isclose(f["krOG"][0](1.0 - 0.20), 0.55), \
            "would mean Swcon was dropped"
        print("ok test_ignoring_connate_water_would_give_a_different_curve")

    def test_assigning_sgof_before_swof_is_rejected() -> None:
        try:
            assign_sgof({}, [SGOF], Regions())
        except ValueError as error:
            assert "SWOF" in str(error)
            print("ok test_assigning_sgof_before_swof_is_rejected")
            return
        raise AssertionError("SGOF without SWOF must be rejected")

    def test_oil_end_point_subtracts_connate_water() -> None:
        f, reg = _with_swof()
        f = assign_sgof(f, [SGOF], reg)
        assert np.isclose(f["krPts"]["og"][0][1], 1.0 - 0.88 - 0.12)
        print("ok test_oil_end_point_subtracts_connate_water")

    def test_gas_end_points_describe_the_deck_table() -> None:
        f, reg = _with_swof()
        f = assign_sgof(f, [SGOF], reg)
        gas = f["krPts"]["g"][0]
        assert np.isclose(gas[0], 0.00) and np.isclose(gas[2], 0.88)
        assert np.isclose(gas[3], 1.00)
        print("ok test_gas_end_points_describe_the_deck_table")

    def test_swof_points_are_not_overwritten() -> None:
        f, reg = _with_swof()
        f = assign_sgof(f, [SGOF], reg)
        assert set(f["krPts"]) == {"w", "ow", "g", "og"}, set(f["krPts"])
        print("ok test_swof_points_are_not_overwritten")

    def test_no_capillary_pressure_field_when_the_deck_states_none() -> None:
        f, reg = _with_swof()
        f = assign_sgof(f, [SGOF], reg)
        assert "pcOG" not in f
        print("ok test_no_capillary_pressure_field_when_the_deck_states_none")

    def test_capillary_pressure_is_attached_when_stated() -> None:
        table = SGOF.copy()
        table[:, 3] = [0.0, 1.0e3, 5.0e3, 2.0e4]
        f, reg = _with_swof()
        f = assign_sgof(f, [table], reg)
        assert "pcOG" in f and np.isclose(f["pcOG"][0](0.20), 1.0e3)
        print("ok test_capillary_pressure_is_attached_when_stated")

    def test_one_curve_per_saturation_region() -> None:
        f, reg = _with_swof(Regions(sat=2))
        f = assign_sgof(f, [SGOF, SGOF], reg)
        assert len(f["krG"]) == 2 and f["krPts"]["g"].shape == (2, 4)
        print("ok test_one_curve_per_saturation_region")

    test_krg_is_read_at_gas_saturation()
    test_krog_accounts_for_connate_water()
    test_ignoring_connate_water_would_give_a_different_curve()
    test_assigning_sgof_before_swof_is_rejected()
    test_oil_end_point_subtracts_connate_water()
    test_gas_end_points_describe_the_deck_table()
    test_swof_points_are_not_overwritten()
    test_no_capillary_pressure_field_when_the_deck_states_none()
    test_capillary_pressure_is_attached_when_stated()
    test_one_curve_per_saturation_region()
    print("\nAll tests passed!")
