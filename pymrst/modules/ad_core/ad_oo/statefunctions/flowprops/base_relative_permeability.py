"""MRST ``ad-core/statefunctions/flowprops/BaseRelativePermeability.m``.

Relative permeability of each active phase. Two-phase systems read each
phase's own curve; three-phase oil interpolates between the oil-water and
oil-gas curves by the water and gas saturations above connate water (MRST's
default Stone-style mixing), unless the fluid gives `krO` directly. Curves may
be tabulated per saturation region and scaled by end points per cell.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.saturation_property import (
    WHOLE,
    SaturationProperty,
)
from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
)
from pymrst.utils.adi import value

CHOP_TOLERANCE: float = 1e-8
CONNATE_OFFSET: float = 1e-5


class BaseRelativePermeability(StateFunction, SaturationProperty):
    """`kr_alpha(s)` for every active phase."""

    def __init__(self, model: Any, regions: Any = None, *,
                 scalingActive: bool = False, relpermPoints: int = 2,
                 immobileChop: bool = False):
        StateFunction.__init__(self, model, label="k_\\alpha", regions=regions)
        self.AutoDiffBackend = model.AutoDiffBackend
        self.scalingActive = scalingActive
        self.relpermPoints = relpermPoints
        self.immobileChop = immobileChop
        self.regions_imbibition = None
        self.depends_on(["s"], STATE)
        if model.water and model.oil and getattr(model.fluid, "krO", None) is None:
            self.store_connate_water(model)
        if self.scalingActive:
            self.store_scalers(model)

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        nph = int(model.water) + int(model.oil) + int(model.gas)
        if nph < 2:
            return [np.ones(model.G.cells.num)]
        if model.oil:
            if model.gas and model.water:
                return self.rel_perm_wog(model, state)
            if model.gas:
                return self.rel_perm_og(model, state)
            return self.rel_perm_wo(model, state)
        if model.gas and model.water:
            return self.rel_perm_wg(model, state)
        return self.rel_perm_unified(model, state)

    def rel_perm_wog(self, model: Any, state: Any) -> List[Any]:
        sw, so, sg = model.get_props(state, "sw", "so", "sg")
        kr_w = self.evaluate_phase_relative_permeability(model, "w", sw)
        kr_g = self.evaluate_gas_rel_perm(model, sg, state)
        if getattr(model.fluid, "krO", None) is not None:
            kr_o = self.evaluate_fluid(model, "krO", so)
        else:
            swcon = self.get_connate_water(model, state)
            swcon = np.minimum(swcon, np.asarray(value(sw)) - CONNATE_OFFSET)
            d = sg + sw - swcon
            ww = (sw - swcon)/d
            krow = self.evaluate_phase_relative_permeability(model, "ow", so)
            krog = self.evaluate_phase_relative_permeability(model, "og", so)
            kr_o = (1 - ww)*krog + ww*krow
        return [kr_w, kr_o, kr_g]

    def rel_perm_wo(self, model: Any, state: Any) -> List[Any]:
        sw, so = model.get_props(state, "sw", "so")
        kr_w = self.evaluate_phase_relative_permeability(model, "w", sw)
        if getattr(model.fluid, "krO", None) is not None:
            kr_o = self.evaluate_fluid(model, "krO", so)
        else:
            kr_o = self.evaluate_phase_relative_permeability(model, "ow", so)
        return [kr_w, kr_o]

    def rel_perm_og(self, model: Any, state: Any) -> List[Any]:
        sg, so = model.get_props(state, "sg", "so")
        kr_g = self.evaluate_gas_rel_perm(model, sg, state)
        if getattr(model.fluid, "krO", None) is not None:
            kr_o = self.evaluate_fluid(model, "krO", so)
        else:
            kr_o = self.evaluate_phase_relative_permeability(model, "og", so)
        return [kr_o, kr_g]

    def rel_perm_wg(self, model: Any, state: Any) -> List[Any]:
        sw, sg = model.get_props(state, "sw", "sg")
        kr_w = self.evaluate_phase_relative_permeability(model, "w", sw)
        kr_g = self.evaluate_gas_rel_perm(model, sg, state)
        return [kr_w, kr_g]

    def rel_perm_unified(self, model: Any, state: Any) -> List[Any]:
        phases = model.get_phase_names()
        s = model.get_props(state, *[f"s{p}" for p in phases])
        s = s if isinstance(s, list) else [s]
        return [self.evaluate_phase_relative_permeability(model, p, s[i])
                for i, p in enumerate(phases)]

    def evaluate_phase_relative_permeability(self, model: Any, phase: str,
                                             s: Any, cells: Any = WHOLE) -> Any:
        fn = getattr(model.fluid, f"kr{phase.upper()}")
        subset = None if isinstance(cells, str) else cells
        if self.scalingActive:
            ss, kr_max_m = self.scale_saturation(model, s, phase, cells,
                                                 "drainage")
            return kr_max_m*self.evaluate_function_cell_subset(fn, subset, ss)
        return self.evaluate_function_cell_subset(fn, subset, s)

    def scale_saturation(self, model: Any, s: Any, phase: str,
                         cells: Any = WHOLE, scale_type: str = "drainage"):
        if isinstance(cells, str):
            cells = self.cell_subset
        elif not (isinstance(self.cell_subset, str)):
            cells = np.asarray(cells)[self.cell_subset]
        sv = np.asarray(value(s))
        scaler = self.get_scalers(model, phase, cells, scale_type)
        p, c, m = scaler["p"], scaler["c"], scaler["m"]
        if len(p) == 2:
            ix1 = sv < p[0]
            ix2 = sv >= p[1]
            ix = ~(ix1 | ix2)
            s_scale = (ix*m)*s + (ix*c + ix2)
        elif len(p) == 3:
            ix1 = (sv >= p[0]) & (sv < p[1])
            ix2 = (sv >= p[1]) & (sv < p[2])
            ix3 = sv >= p[2]
            a = ix1*m[0] + ix2*m[1]
            s_scale = a*s + ix1*c[0] + ix2*c[1] + ix3
        else:
            raise ValueError("Unknown number of scaling points")
        return s_scale, scaler["k_max_m"]

    def apply_immobile_chop(self, model: Any, state: Any, state0: Any):
        """Snap saturations that cross a critical point to just past it.

        Off by default, as in MRST. Returns `(state, chopped_rows)`.
        """
        chopped: Any = np.zeros(model.G.cells.num, dtype=bool)
        if not self.immobileChop:
            return state, chopped
        s_min = self.get_critical_phase_saturations(model, state)
        s = np.array(state["s"], dtype=float, copy=True)
        s0 = np.asarray(state0["s"], dtype=float)
        tol = CHOP_TOLERANCE
        for ph in range(s.shape[1]):
            sm = s_min[0, ph]
            to_mobile = (s0[:, ph] < sm + tol) & (s[:, ph] > sm - tol)
            to_immobile = (s0[:, ph] > sm - tol) & (s[:, ph] < sm + tol)
            s[to_immobile, ph] = sm - 2*tol
            s[to_mobile, ph] = sm + 2*tol
            chopped = chopped | to_immobile | to_mobile
        state["s"] = s
        return state, chopped

    def subset(self, subs: Any) -> "BaseRelativePermeability":
        out = super().subset(subs)
        out.cell_subset = subs
        return out

    def evaluate_gas_rel_perm(self, model: Any, sg: Any, state: Any) -> Any:
        return self.evaluate_phase_relative_permeability(model, "g", sg)

    evaluateOnDomain = evaluate_on_domain
    relPermWOG = rel_perm_wog
    relPermWO = rel_perm_wo
    relPermOG = rel_perm_og
    relPermWG = rel_perm_wg
    relPermUnified = rel_perm_unified
    evaluatePhaseRelativePermeability = evaluate_phase_relative_permeability
    scaleSaturation = scale_saturation
    applyImmobileChop = apply_immobile_chop
    evaluateGasRelPerm = evaluate_gas_rel_perm


__all__ = ["BaseRelativePermeability"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_two_phase_curves_are_each_phases_own():
        model = toy_model("WO", nx=2)
        state = toy_state(model, s=np.array([[0.2, 0.8], [0.6, 0.4]]))
        kr_w, kr_o = model.get_prop(state, "RelativePermeability")
        np.testing.assert_allclose(kr_w, [0.04, 0.36])
        np.testing.assert_allclose(kr_o, [0.64, 0.16])
        print("ok test_two_phase_curves_are_each_phases_own")

    def test_three_phase_oil_mixes_the_two_oil_curves():
        model = toy_model("WOG", nx=1)
        state = toy_state(model, s=np.array([[0.3, 0.5, 0.2]]))
        _, kr_o, _ = model.get_prop(state, "RelativePermeability")
        # swcon = 0 for this fluid, so the weight on krow is sw/(sw + sg).
        ww = 0.3/0.5
        np.testing.assert_allclose(kr_o, (1 - ww)*0.25 + ww*0.25)
        print("ok test_three_phase_oil_mixes_the_two_oil_curves")

    def test_single_phase_relperm_is_one():
        model = toy_model("W", nx=3)
        (kr,) = model.get_prop(toy_state(model), "RelativePermeability")
        np.testing.assert_allclose(kr, 1.0)
        print("ok test_single_phase_relperm_is_one")

    def test_chop_snaps_a_crossing_to_just_past_the_critical_point():
        model = toy_model("WO", nx=2, smin=[0.2, 0.1])
        prop = model.FlowPropertyFunctions.get_state_function("RelativePermeability")
        prop.immobileChop = True
        state0 = toy_state(model, s=np.array([[0.1, 0.9], [0.5, 0.5]]))
        state = toy_state(model, s=np.array([[0.3, 0.7], [0.5, 0.5]]))
        state, chopped = prop.apply_immobile_chop(model, state, state0)
        np.testing.assert_allclose(state["s"][0, 0], 0.2 + 2*CHOP_TOLERANCE)
        np.testing.assert_array_equal(chopped, [True, False])
        print("ok test_chop_snaps_a_crossing_to_just_past_the_critical_point")

    test_two_phase_curves_are_each_phases_own()
    test_three_phase_oil_mixes_the_two_oil_curves()
    test_single_phase_relperm_is_one()
    test_chop_snaps_a_crossing_to_just_past_the_critical_point()
    print("\nAll tests passed!")
