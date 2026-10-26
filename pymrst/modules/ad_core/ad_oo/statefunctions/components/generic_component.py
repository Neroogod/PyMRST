"""MRST ``ad-core/statefunctions/components/GenericComponent.m``.

Base class for a conserved component in the generic (mass-based) models. A
component reports how much of itself each phase carries per unit volume
(`get_component_density`); mass, mobility and composition follow from that
and the model's saturations, pore volume, mobility and density.
"""

from __future__ import annotations

from typing import Any, List, Optional

from pymrst.modules.ad_core.ad_oo.statefunctions.state_function_dependent import (
    StateFunctionDependent,
)


class GenericComponent(StateFunctionDependent):
    """A named component; subclasses say which phases hold it."""

    def __init__(self, name: str):
        self.name = name
        self.externals = None
        self.molarMass = 1.0
        self.function_depends_on("getComponentMobility", ["Mobility"],
                                 "FlowPropertyFunctions")
        self.function_depends_on("getComponentMobility", ["PoreVolume", "Density"],
                                 "PVTPropertyFunctions")
        self.function_depends_on("getComponentMass", ["PoreVolume", "Density"],
                                 "PVTPropertyFunctions")
        self.function_depends_on("getComponentMass", "s", "state")

    def get_phase_composition(self, model: Any, state: Any, *args: Any) -> List[Any]:
        """Mass fraction of this component in each phase."""
        c = self.get_component_density(model, state)
        rho = model.get_prop(state, "Density")
        rho = rho if isinstance(rho, list) else [rho]
        return [None if ci is None else ci/rho[ph] for ph, ci in enumerate(c)]

    def get_component_mass(self, model: Any, state: Any) -> List[Optional[Any]]:
        s = _columns(model.get_prop(state, "s"))
        pv = model.PVTPropertyFunctions.get(model, state, "PoreVolume")
        mass = self.get_component_density(model, state)
        return [None if m is None else s[i]*pv*m for i, m in enumerate(mass)]

    def get_component_mobility(self, model: Any, state: Any, *args: Any) -> List[Any]:
        mob = model.FlowPropertyFunctions.get(model, state, "Mobility")
        density = self.get_component_density(model, state, *args)
        return [None if d is None else mob[i]*d for i, d in enumerate(density)]

    def get_component_density(self, model: Any, state: Any, *args: Any) -> List[Any]:
        return [None]*model.get_number_of_phases()

    def get_pure_phase_density_surface(self, model, state, pressure, temperature):
        return [None]*model.get_number_of_phases()

    def get_phase_composition_surface(self, model, state, pressure, temperature):
        return [None]*model.get_number_of_phases()

    def get_phase_component_fraction_injection(self, model, state, force):
        return [None]*model.get_number_of_phases()

    getPhaseComposition = get_phase_composition
    getComponentMass = get_component_mass
    getComponentMobility = get_component_mobility
    getComponentDensity = get_component_density
    getPurePhaseDensitySurface = get_pure_phase_density_surface
    getPhaseCompositionSurface = get_phase_composition_surface
    getPhaseComponentFractionInjection = get_phase_component_fraction_injection


def _columns(s: Any) -> List[Any]:
    """MRST `expandMatrixToCell`: a saturation matrix as a list of columns."""
    if isinstance(s, list):
        return s
    import numpy as np
    s = np.asarray(s)
    return [s[:, i] for i in range(s.shape[1])] if s.ndim > 1 else [s]


__all__ = ["GenericComponent"]


if __name__ == "__main__":
    def test_the_generic_component_occupies_no_phase():
        from pymrst.modules.ad_core.ad_oo.testing import toy_model, toy_state
        model = toy_model("WO", nx=2)
        component = GenericComponent("tracer")
        assert component.get_component_density(model, toy_state(model)) == [None, None]
        assert component.get_component_mass(model, toy_state(model)) == [None, None]
        print("ok test_the_generic_component_occupies_no_phase")

    def test_mass_depends_on_pore_volume_density_and_saturation():
        dep = GenericComponent("x").get_function_dependencies("getComponentMass")
        assert {(d.name, d.grouping) for d in dep["externals"]} == {
            ("PoreVolume", "PVTPropertyFunctions"), ("Density", "PVTPropertyFunctions"),
            ("s", "state")}
        print("ok test_mass_depends_on_pore_volume_density_and_saturation")

    test_the_generic_component_occupies_no_phase()
    test_mass_depends_on_pore_volume_density_and_saturation()
    print("\nAll tests passed!")
