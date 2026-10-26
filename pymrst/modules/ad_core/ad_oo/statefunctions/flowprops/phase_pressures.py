"""MRST ``ad-core/statefunctions/flowprops/PhasePressures.m``.

Each phase's pressure: the reference (oil) pressure plus that phase's
capillary pressure, when the fluid has any.
"""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.statefunctions.state_function import (
    STATE,
    StateFunction,
)


class PhasePressures(StateFunction):
    """`p_alpha = p + pc_alpha`."""

    def __init__(self, model: Any, regions: Any = None):
        super().__init__(model, label="p_\\alpha", regions=regions)
        self.AutoDiffBackend = model.AutoDiffBackend
        f = model.fluid
        has = lambda name: getattr(f, name, None) is not None
        self.has_pc = ((model.water and model.oil and has("pcOW"))
                       or (model.gas and model.oil and has("pcOG"))
                       or (not model.oil and has("pcWG")))
        if self.has_pc:
            self.depends_on(["CapillaryPressure"], "FlowPropertyFunctions")
        self.depends_on(["pressure"], STATE)

    def evaluate_on_domain(self, model: Any, state: Any) -> List[Any]:
        p = model.get_props(state, "Pressure")
        nph = model.get_number_of_phases()
        p_phase = [p]*nph
        if self.has_pc:
            pc = self.get_evaluated_externals(model, state, "CapillaryPressure")
            for i in range(nph):
                if pc[i] is not None:
                    p_phase[i] = p_phase[i] + pc[i]
        return p_phase

    evaluateOnDomain = evaluate_on_domain


__all__ = ["PhasePressures"]


if __name__ == "__main__":
    import numpy as np

    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_without_capillarity_every_phase_has_the_reference_pressure():
        model = toy_model("WO", nx=2)
        state = toy_state(model)
        p_phase = model.get_prop(state, "PhasePressures")
        for p in p_phase:
            np.testing.assert_allclose(p, state["pressure"])
        print("ok test_without_capillarity_every_phase_has_the_reference_pressure")

    def test_water_pressure_is_oil_pressure_less_pcow():
        model = toy_model("WO", nx=2, pc_ow=lambda sw: 1e5*(1.0 - sw))
        state = toy_state(model, s=np.array([[0.2, 0.8], [0.6, 0.4]]))
        p_w, p_o = model.get_prop(state, "PhasePressures")
        np.testing.assert_allclose(p_o, state["pressure"])
        np.testing.assert_allclose(p_w, state["pressure"] - 1e5*np.array([0.8, 0.4]))
        print("ok test_water_pressure_is_oil_pressure_less_pcow")

    test_without_capillarity_every_phase_has_the_reference_pressure()
    test_water_pressure_is_oil_pressure_less_pcow()
    print("\nAll tests passed!")
