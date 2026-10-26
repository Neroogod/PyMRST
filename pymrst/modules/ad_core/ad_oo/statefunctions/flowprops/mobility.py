"""MRST ``ad-core/statefunctions/flowprops/Mobility.m``.

Phase mobility `kr / mu`, times the fluid's pressure-dependent transmissibility
multiplier `tranMultR(p)` when it has one.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class Mobility(StateFunction):
    """`lambda_alpha = kr_alpha / mu_alpha`."""

    def __init__(self, model: Any, regions: Any = None):
        super().__init__(model, label="\\lambda_\\alpha", regions=regions)
        self.AutoDiffBackend = model.AutoDiffBackend
        self.hasMultipliers = False
        self.depends_on(["RelativePermeability"])
        self.depends_on("Viscosity", "PVTPropertyFunctions")
        self.output_range = (0.0, np.inf)

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        kr = self.get_evaluated_dependencies(state, "RelativePermeability")
        mu = self.get_evaluated_externals(model, state, "Viscosity")
        mob = [k/m for k, m in zip(kr, mu)]
        if getattr(model.fluid, "tranMultR", None) is not None:
            p = model.get_prop(state, "pressure")
            mult = model.fluid.tranMultR(p)
            mob = [m*mult for m in mob]
        if self.hasMultipliers:
            # MRST multiplies `mu` here, after `mob` is already formed, so
            # the multipliers never reach the result. Ported as written.
            multipliers = self.get_evaluated_dependencies(state,
                                                          "MobilityMultipliers")
            for i, m in enumerate(multipliers):
                if m is not None:
                    mu[i] = mu[i]*m
        is_ad = [hasattr(m, "jac") for m in mob]
        if any(is_ad) and not all(is_ad):
            sample = mob[is_ad.index(True)]
            mob = [m if hasattr(m, "jac") else
                   model.AutoDiffBackend.convert_to_ad(m, sample) for m in mob]
        return mob

    def enable_multipliers(self) -> "Mobility":
        self.hasMultipliers = True
        self.depends_on("MobilityMultipliers")
        return self

    evaluateOnDomain = evaluate_on_domain
    enableMultipliers = enable_multipliers


__all__ = ["Mobility"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_mobility_is_relperm_over_viscosity():
        model = toy_model("WO", nx=2)
        state = toy_state(model, s=np.array([[0.2, 0.8], [0.5, 0.5]]))
        mob_w, mob_o = model.get_prop(state, "Mobility")
        np.testing.assert_allclose(mob_w, np.array([0.04, 0.25])/1e-3)
        np.testing.assert_allclose(mob_o, np.array([0.64, 0.25])/5e-3)
        print("ok test_mobility_is_relperm_over_viscosity")

    def test_pressure_multiplier_scales_every_phase():
        model = toy_model("WO", nx=2)
        model.fluid.tranMultR = lambda p: 2.0 + 0.0*p
        state = toy_state(model, s=np.array([[0.5, 0.5], [0.5, 0.5]]))
        mob_w, _ = model.get_prop(state, "Mobility")
        np.testing.assert_allclose(mob_w, 2.0*0.25/1e-3)
        print("ok test_pressure_multiplier_scales_every_phase")

    test_mobility_is_relperm_over_viscosity()
    test_pressure_multiplier_scales_every_phase()
    print("\nAll tests passed!")
