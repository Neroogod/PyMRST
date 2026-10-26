"""MRST ``ad-core/models/facilities/setupWellControlEquationsSingleWell.m``.

The one control equation of a standard well: its bottom-hole pressure, its
total or single-phase surface rate, or a tubing-head pressure through a VFP
table, equal to the target. A shut well pins its bottom-hole pressure, and a
rate control with a zero target closes on the *total* rate - a single-phase
zero target would only duplicate that phase's well equation.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from pymrst.utils.units import BARSA


def setup_well_control_equations_single_well(well: Any, sol0: Any, sol: Any,
                                             pBH: Any, q_s: Sequence[Any],
                                             status: bool, mix_s: np.ndarray,
                                             model: Any) -> Any:
    act_ph = model.get_active_phases()
    n_ph = len(q_s)
    iw = 0 if model.water else None
    io = int(np.sum(act_ph[:2])) - 1 if model.oil else None
    ig = int(np.sum(act_ph[:3])) - 1 if model.gas else None
    ctrl_type = _get(sol, "type")
    val = _get(sol, "val")
    bhp = _get(sol, "bhp")
    qt_s = 0.0
    for ph in range(n_ph):
        qt_s = qt_s + q_s[ph]
    set_to_zero_rate = val == 0 and ctrl_type.lower() != "bhp"
    mix_s = np.sum(np.abs(np.atleast_2d(mix_s)), axis=0)
    if not status:
        return (pBH - bhp)/BARSA
    if set_to_zero_rate:
        return qt_s
    if ctrl_type == "bhp":
        return pBH - val
    if ctrl_type in ("rate", "vrat"):
        return qt_s - val
    if ctrl_type == "orat":
        assert model.oil, "Oil phase must be present to control wells on oil rates"
        return qt_s if mix_s[io] == 0 else q_s[io] - val
    if ctrl_type == "wrat":
        assert model.water, "Water phase must be present to control wells on oil rates"
        return qt_s if mix_s[iw] == 0 else q_s[iw] - val
    if ctrl_type == "grat":
        assert model.gas, "Gas phase must be present to control wells on oil rates"
        return qt_s if mix_s[ig] == 0 else q_s[ig] - val
    if ctrl_type == "lrat":
        if mix_s[iw] + mix_s[io] == 0:
            return qt_s
        return q_s[iw] + q_s[io] - val
    if ctrl_type == "thp":
        return _thp_equation(well, val, pBH, q_s, model, iw, io, ig)
    raise ValueError(f"Unknown well control {ctrl_type} for well {_get(sol, 'name')}")


def _thp_equation(well, val, pBH, q_s, model, iw, io, ig):
    vfp = well.VFPTable
    flow_type = vfp.flowType.lower()
    if flow_type == "oil":
        flow = q_s[io]
    elif flow_type == "gas":
        flow = q_s[ig]
    elif flow_type == "wat":
        flow = q_s[iw]
    elif flow_type == "liq":
        # MRST sums oil and *gas* for a liquid flow type, as written.
        flow = 0.0
        if model.oil:
            flow = flow + q_s[io]
        if model.gas:
            flow = flow + q_s[ig]
    else:
        raise ValueError(f"Unknown VFP flow type: '{vfp.flowType}'")
    if well.is_injector():
        bhp = vfp.evaluate_bhp(flow, val)
    else:
        gas_ratio = vfp.gasRatioType.lower()
        if gas_ratio == "gor":
            gr = q_s[ig]/q_s[io]
        elif gas_ratio == "glr":
            liquid = 0.0
            if model.water:
                liquid = liquid + q_s[iw]
            if model.oil:
                liquid = liquid + q_s[io]
            gr = q_s[ig]/liquid
        elif gas_ratio == "ogr":
            gr = q_s[io]/q_s[ig]
        else:
            raise ValueError("Unsupported gas ratio type for VFP")
        water_ratio = vfp.waterRatioType.lower()
        if water_ratio == "wor":
            wr = q_s[iw]/q_s[io]
        elif water_ratio == "wct":
            liquid = q_s[iw]
            if model.oil:
                liquid = liquid + q_s[io]
            wr = q_s[iw]/liquid
        elif water_ratio == "wgr":
            wr = q_s[iw]/q_s[ig]
        else:
            raise ValueError("Unsupported water ratio type for VFP")
        bhp = vfp.evaluate_bhp(flow, val, wr, gr)
    return bhp - pBH


def _get(container: Any, name: str) -> Any:
    if isinstance(container, dict):
        return container[name]
    return getattr(container, name)


setupWellControlEquationsSingleWell = setup_well_control_equations_single_well

__all__ = ["setup_well_control_equations_single_well",
           "setupWellControlEquationsSingleWell"]


if __name__ == "__main__":
    from types import SimpleNamespace

    class _Model:
        water = oil = True
        gas = False

        def get_active_phases(self):
            return np.array([True, True, False])

    def _sol(ctrl_type, val, bhp=100e5):
        return SimpleNamespace(type=ctrl_type, val=val, bhp=bhp, name="P1")

    Q = [np.array([-1.0]), np.array([-3.0])]
    MIX = np.array([0.25, 0.75])

    def test_bhp_control_is_pressure_minus_target():
        eq = setup_well_control_equations_single_well(
            None, None, _sol("bhp", 90e5), np.array([95e5]), Q, True, MIX, _Model())
        np.testing.assert_allclose(eq, 5e5)
        print("ok test_bhp_control_is_pressure_minus_target")

    def test_oil_rate_control_reads_the_oil_rate():
        eq = setup_well_control_equations_single_well(
            None, None, _sol("orat", -2.0), np.array([95e5]), Q, True, MIX, _Model())
        np.testing.assert_allclose(eq, -1.0)
        print("ok test_oil_rate_control_reads_the_oil_rate")

    def test_zero_target_closes_on_total_rate():
        eq = setup_well_control_equations_single_well(
            None, None, _sol("orat", 0.0), np.array([95e5]), Q, True, MIX, _Model())
        np.testing.assert_allclose(eq, -4.0)
        print("ok test_zero_target_closes_on_total_rate")

    def test_a_shut_well_pins_its_bottom_hole_pressure_in_bar():
        eq = setup_well_control_equations_single_well(
            None, None, _sol("orat", -2.0, bhp=100e5), np.array([101e5]), Q,
            False, MIX, _Model())
        np.testing.assert_allclose(eq, 1.0)
        print("ok test_a_shut_well_pins_its_bottom_hole_pressure_in_bar")

    def test_an_unknown_control_is_refused():
        try:
            setup_well_control_equations_single_well(
                None, None, _sol("xyz", -2.0), np.array([95e5]), Q, True, MIX,
                _Model())
            raise SystemExit("should have raised")
        except ValueError:
            pass
        print("ok test_an_unknown_control_is_refused")

    test_bhp_control_is_pressure_minus_target()
    test_oil_rate_control_reads_the_oil_rate()
    test_zero_target_closes_on_total_rate()
    test_a_shut_well_pins_its_bottom_hole_pressure_in_bar()
    test_an_unknown_control_is_refused()
    print("\nAll tests passed!")
