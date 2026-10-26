"""MRST ``ad-core/statefunctions/components/ImmiscibleComponent.m``.

A component that lives in exactly one phase and makes up all of it: its
density in that phase is the phase density, and its composition is 1.
`phaseIndex` is 0-based here (MRST's is 1-based).
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from pymrst.modules.ad_core.ad_oo.statefunctions.components.generic_component import (
    GenericComponent,
)


class ImmiscibleComponent(GenericComponent):
    """All of one phase, none of the others."""

    def __init__(self, name: str, phase: int):
        super().__init__(name)
        self.phaseIndex = phase
        self.function_depends_on("getComponentDensity", "Density",
                                 "PVTPropertyFunctions")

    def get_component_density(self, model: Any, state: Any, *args: Any) -> List[Any]:
        rho = model.PVTPropertyFunctions.get(model, state, "Density")
        c = super().get_component_density(model, state)
        if len(c) == 1 and not isinstance(rho, list):
            c[0] = rho
        else:
            c[self.phaseIndex] = rho[self.phaseIndex]
        return c

    def get_phase_composition(self, model: Any, state: Any, *args: Any) -> List[Any]:
        c: List[Any] = [None]*model.get_number_of_phases()
        c[self.phaseIndex] = 1
        return c

    def get_phase_composition_surface(self, model, state, pressure, temperature):
        c: List[Any] = [None]*model.get_number_of_phases()
        c[self.phaseIndex] = 1
        return c

    def get_phase_component_fraction_injection(self, model: Any, state: Any,
                                               force: Any) -> List[Any]:
        """This component's share of what each well or boundary injects."""
        c: List[Any] = [None]*model.get_number_of_phases()
        force = force if isinstance(force, (list, tuple)) else [force]
        source = "compi" if all(getattr(f, "compi", None) is not None for f in force) else "sat"
        comp_i = np.vstack([np.atleast_2d(np.asarray(getattr(f, source), dtype=float))
                            for f in force])
        ci = comp_i[:, self.phaseIndex]
        if np.any(ci != 0):
            c[self.phaseIndex] = ci
        return c

    getComponentDensity = get_component_density
    getPhaseComposition = get_phase_composition
    getPhaseCompositionSurface = get_phase_composition_surface
    getPhaseComponentFractionInjection = get_phase_component_fraction_injection


__all__ = ["ImmiscibleComponent"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state

    def test_density_is_the_phase_density_in_its_own_phase_only():
        model = toy_model("WO", nx=2)
        state = toy_state(model)
        c = ImmiscibleComponent("oil", 1).get_component_density(model, state)
        assert c[0] is None
        np.testing.assert_allclose(c[1], model.get_prop(state, "Density")[1])
        print("ok test_density_is_the_phase_density_in_its_own_phase_only")

    def test_injection_share_comes_from_the_well_composition():
        model = toy_model("WO", nx=2)
        wells = [SimpleNamespace(compi=np.array([1.0, 0.0])),
                 SimpleNamespace(compi=np.array([0.0, 1.0]))]
        c = ImmiscibleComponent("water", 0).get_phase_component_fraction_injection(
            model, None, wells)
        np.testing.assert_allclose(c[0], [1.0, 0.0])
        assert c[1] is None
        print("ok test_injection_share_comes_from_the_well_composition")

    test_density_is_the_phase_density_in_its_own_phase_only()
    test_injection_share_comes_from_the_well_composition()
    print("\nAll tests passed!")
