"""MRST ``ad-core/statefunctions/flowprops/PressureReductionFactors.m``.

Weights that combine the phase equations into a pressure equation (true
IMPES): `pv / rho_alpha` for each phase.
"""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class PressureReductionFactors(StateFunction):
    """`w_alpha^p = pv / rho_alpha`."""

    def __init__(self, model: Any, regions: Any = None):
        super().__init__(model, label="w_i^p", regions=regions)
        self.AutoDiffBackend = model.AutoDiffBackend
        self.depends_on(["Density", "PoreVolume"])

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        rho, pv = self.get_evaluated_dependencies(state, "Density", "PoreVolume")
        return [pv/rho[ph] for ph in range(len(rho))]

    evaluateOnDomain = evaluate_on_domain


__all__ = ["PressureReductionFactors"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_weights_are_pore_volume_over_density():
        model = toy_model("WO", nx=2)
        state = toy_state(model)
        w = model.get_prop(state, "PressureReductionFactors")
        rho = model.get_prop(state, "Density")
        for w_i, rho_i in zip(w, rho):
            np.testing.assert_allclose(w_i, 0.3/rho_i)
        print("ok test_weights_are_pore_volume_over_density")

    test_weights_are_pore_volume_over_density()
    print("\nAll tests passed!")
