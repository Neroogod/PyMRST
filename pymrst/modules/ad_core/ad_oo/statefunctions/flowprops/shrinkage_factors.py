"""MRST ``ad-core/statefunctions/flowprops/ShrinkageFactors.m``.

`b_alpha = 1/B_alpha` per phase, each at its own phase pressure, from the
fluid's `bW`, `bO`, `bG`.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
)
from pymrst.modules.ad_core.utils.get_sample_ad import get_sample_ad


class ShrinkageFactors(StateFunction):
    """Pressure-dependent shrinkage factors for every active phase."""

    usePhasePressures = True

    def __init__(self, model: Any, regions: Any = None):
        super().__init__(model, label="b_\\alpha", regions=regions)
        self.AutoDiffBackend = model.AutoDiffBackend
        if self.usePhasePressures:
            self.depends_on("PhasePressures")
        else:
            self.depends_on("pressure", STATE)
        self.output_range = (0.0, np.inf)

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        names = model.get_phase_names()
        nph = len(names)
        if self.usePhasePressures:
            p_phase = self.get_evaluated_dependencies(state, "PhasePressures")
            p_phase = p_phase if isinstance(p_phase, list) else [p_phase]
        else:
            p_phase = [model.get_prop(state, "pressure")]*nph
        sample, is_ad = get_sample_ad(*p_phase)
        b = [self.evaluate_phase_shrinkage_factor(model, state, names[ph],
                                                  p_phase[ph])
             for ph in range(nph)]
        if is_ad:
            b = [x if hasattr(x, "jac") else
                 model.AutoDiffBackend.convert_to_ad(x, sample) for x in b]
        return b

    def evaluate_phase_shrinkage_factor(self, model: Any, state: Any,
                                        name: str, p: Any) -> Any:
        return self.evaluate_fluid(model, f"b{name}", p)

    evaluateOnDomain = evaluate_on_domain
    evaluatePhaseShrinkageFactor = evaluate_phase_shrinkage_factor


__all__ = ["ShrinkageFactors"]


if __name__ == "__main__":
    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state
    from pymrst.utils.adi import init_variables_adi

    def test_each_phase_uses_its_own_curve():
        model = toy_model("WO", nx=2)
        state = toy_state(model)
        b_w, b_o = model.get_prop(state, "ShrinkageFactors")
        np.testing.assert_allclose(b_w, model.fluid.bW(state["pressure"]))
        np.testing.assert_allclose(b_o, model.fluid.bO(state["pressure"]))
        print("ok test_each_phase_uses_its_own_curve")

    def test_derivatives_follow_pressure():
        model = toy_model("WO", nx=2)
        state = toy_state(model)
        (state["pressure"],) = init_variables_adi(state["pressure"])
        b = model.get_prop(state, "ShrinkageFactors")
        assert all(hasattr(x, "jac") for x in b)
        print("ok test_derivatives_follow_pressure")

    test_each_phase_uses_its_own_curve()
    test_derivatives_follow_pressure()
    print("\nAll tests passed!")
