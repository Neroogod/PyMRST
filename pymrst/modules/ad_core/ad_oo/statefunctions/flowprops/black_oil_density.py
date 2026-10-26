"""MRST ``ad-core/statefunctions/flowprops/BlackOilDensity.m``.

Reservoir phase densities: surface density times shrinkage factor, plus the
mass oil carries as dissolved gas and gas carries as vaporised oil.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
)


class BlackOilDensity(StateFunction):
    """`rho_alpha = b_alpha * rhoS_alpha` (+ `rs`/`rv` carry)."""

    def __init__(self, model: Any, regions: Any = None):
        super().__init__(model, label="\\rho_\\alpha", regions=regions)
        self.AutoDiffBackend = model.AutoDiffBackend
        self.disgas = False
        self.vapoil = False
        if hasattr(model, "disgas"):
            self.disgas = bool(model.disgas)
            if self.disgas:
                self.depends_on(["rs"], STATE)
        if hasattr(model, "vapoil"):
            self.vapoil = bool(model.vapoil)
            if self.vapoil:
                self.depends_on(["rv"], STATE)
        self.depends_on(["ShrinkageFactors", "SurfaceDensity"])
        self.output_range = (0.0, np.inf)

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        b, rho_s = self.get_evaluated_dependencies(state, "ShrinkageFactors",
                                                   "SurfaceDensity")
        rho = [rho_s[i]*b[i] for i in range(len(rho_s))]
        if (self.disgas or self.vapoil) and model.gas and model.oil:
            oix, gix = model.get_phase_index("O", "G")
            if self.disgas:
                rs = model.get_prop(state, "rs")
                rho[oix] = rho[oix] + rs*b[oix]*rho_s[gix]
            if self.vapoil:
                rv = model.get_prop(state, "rv")
                rho[gix] = rho[gix] + rv*b[gix]*rho_s[oix]
        return rho

    evaluateOnDomain = evaluate_on_domain


__all__ = ["BlackOilDensity"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_density_is_shrinkage_times_surface_density():
        model = toy_model("WO", nx=2)
        state = toy_state(model)
        rho_w, rho_o = model.get_prop(state, "Density")
        np.testing.assert_allclose(rho_w, 1000.0*model.fluid.bW(state["pressure"]))
        np.testing.assert_allclose(rho_o, 700.0*model.fluid.bO(state["pressure"]))
        print("ok test_density_is_shrinkage_times_surface_density")

    test_density_is_shrinkage_times_surface_density()
    print("\nAll tests passed!")
