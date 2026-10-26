"""MRST ``ad-core/statefunctions/flowprops/ComponentPhaseMass.m``.

Mass of each component in each phase, `ncomp x nphase`, from every
component's `getComponentMass`.
"""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.component_property import (
    ComponentProperty,
)
from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class ComponentPhaseMass(StateFunction, ComponentProperty):
    """`M_{i,alpha}`; a phase a component does not occupy is `None`."""

    def __init__(self, model: Any, regions: Any = None):
        StateFunction.__init__(self, model, label="M_{i,\\alpha}",
                               regions=regions)
        self.AutoDiffBackend = model.AutoDiffBackend
        self.init_component_property(model, "getComponentMass")

    def evaluate_on_domain(self, model: Any, state: Any) -> List[List[Any]]:
        return [list(component.get_component_mass(model, state))
                for component in model.Components]

    evaluateOnDomain = evaluate_on_domain


__all__ = ["ComponentPhaseMass"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def test_one_row_per_component():
        component = SimpleNamespace(
            get_function_dependencies=lambda name: {},
            get_component_mass=lambda model, state: [1.0, None])
        model = SimpleNamespace(Components=[component]*2,
                                AutoDiffBackend=None)
        prop = ComponentPhaseMass(model)
        assert prop.evaluate_on_domain(model, {}) == [[1.0, None], [1.0, None]]
        print("ok test_one_row_per_component")

    test_one_row_per_component()
    print("\nAll tests passed!")
