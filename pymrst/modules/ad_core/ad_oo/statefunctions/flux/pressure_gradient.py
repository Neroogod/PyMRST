"""MRST ``ad-core/statefunctions/flux/PressureGradient.m``.

Gradient of each phase pressure across every face. Without capillary pressure
all phases share the reference pressure's gradient.
"""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
)


class PressureGradient(StateFunction):
    """`grad(p_alpha)` per phase."""

    def __init__(self, model: Any):
        super().__init__(model, label="\\nabla p_\\alpha")
        self.AutoDiffBackend = model.AutoDiffBackend
        self.depends_on("PhasePressures", "PVTPropertyFunctions")
        self.depends_on("pressure", STATE)
        self.Grad = model.operators.grad

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        nph = int(model.get_active_phases().sum())
        pc = model.FlowPropertyFunctions.get_state_function("CapillaryPressure")
        if pc.pc_present(model):
            p = self.get_evaluated_externals(model, state, "PhasePressures")
            return [self.Grad(p[i]) for i in range(nph)]
        dp = self.Grad(model.get_prop(state, "pressure"))
        return [dp]*nph

    evaluateOnDomain = evaluate_on_domain


__all__ = ["PressureGradient"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_without_capillarity_every_phase_shares_the_gradient():
        model = toy_model("WO", nx=3)
        state = toy_state(model, pressure=np.array([3.0, 2.0, 0.0]))
        dp_w, dp_o = model.get_prop(state, "PressureGradient")
        np.testing.assert_allclose(dp_w, [-1.0, -2.0])
        np.testing.assert_allclose(dp_o, dp_w)
        print("ok test_without_capillarity_every_phase_shares_the_gradient")

    def test_with_capillarity_water_follows_its_own_pressure():
        model = toy_model("WO", nx=2, pc_ow=lambda sw: 10.0*sw)
        state = toy_state(model, pressure=np.array([5.0, 5.0]),
                          s=np.array([[0.2, 0.8], [0.6, 0.4]]))
        dp_w, dp_o = model.get_prop(state, "PressureGradient")
        np.testing.assert_allclose(dp_o, [0.0])
        np.testing.assert_allclose(dp_w, [-(6.0 - 2.0)])
        print("ok test_with_capillarity_water_follows_its_own_pressure")

    test_without_capillarity_every_phase_shares_the_gradient()
    test_with_capillarity_water_follows_its_own_pressure()
    print("\nAll tests passed!")
