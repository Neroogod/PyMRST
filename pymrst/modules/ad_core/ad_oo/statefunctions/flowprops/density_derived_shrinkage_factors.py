"""MRST ``ad-core/statefunctions/flowprops/DensityDerivedShrinkageFactors.m``:
shrinkage factors from the phase densities, `b_alpha = rho_alpha/rho_alpha^S`."""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class DensityDerivedShrinkageFactors(StateFunction):
    """`b_alpha` per phase, from `Density` and the surface densities."""

    def __init__(self, model: Any, *args: Any, **kwargs: Any):
        super().__init__(model, *args, label="b_\\alpha", **kwargs)
        self.depends_on(["Density"])

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        b = list(self.get_evaluated_dependencies(state, "Density"))
        rhoS = np.asarray(model.get_surface_densities(), dtype=float).ravel()
        for i in range(len(b)):
            b[i] = b[i]/rhoS[i]
        return b

    evaluateOnDomain = evaluate_on_domain


__all__ = ["DensityDerivedShrinkageFactors"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_density_over_surface_density_recovers_b():
        """Registered next to MRST's `ShrinkageFactors`, it gives the same b."""
        model = toy_model("WO", nx=3)
        model.PVTPropertyFunctions.set_state_function("DerivedB", DensityDerivedShrinkageFactors(model))
        state = toy_state(model)
        derived = model.get_prop(state, "DerivedB")
        b = model.get_prop(state, "ShrinkageFactors")
        for d, ref in zip(derived, b):
            np.testing.assert_allclose(d, ref, rtol=1e-14)
        print("ok test_density_over_surface_density_recovers_b")

    test_density_over_surface_density_recovers_b()
    print("\nAll tests passed!")
