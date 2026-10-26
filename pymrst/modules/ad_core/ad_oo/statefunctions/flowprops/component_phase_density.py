"""MRST ``ad-core/statefunctions/flowprops/ComponentPhaseDensity.m``.

Mass of each component per unit volume of each phase, from every component's
`getComponentDensity`.
"""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.component_property import (
    ComponentProperty,
)
from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class ComponentPhaseDensity(StateFunction, ComponentProperty):
    """`rho_alpha x_{i,alpha}`, `ncomp x nphase`."""

    def __init__(self, model: Any, regions: Any = None):
        StateFunction.__init__(self, model, label="\\rho_\\alpha x_{i,\\alpha}",
                               regions=regions)
        self.AutoDiffBackend = model.AutoDiffBackend
        self.init_component_property(model, "getComponentDensity")

    def evaluate_on_domain(self, model: Any, state: Any) -> List[List[Any]]:
        return [list(component.get_component_density(model, state))
                for component in model.Components]

    evaluateOnDomain = evaluate_on_domain


__all__ = ["ComponentPhaseDensity"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def test_one_row_per_component():
        component = SimpleNamespace(
            get_function_dependencies=lambda name: {},
            get_component_density=lambda model, state: [None, 3.0])
        model = SimpleNamespace(Components=[component], AutoDiffBackend=None)
        assert ComponentPhaseDensity(model).evaluate_on_domain(model, {}) == [[None, 3.0]]
        print("ok test_one_row_per_component")

    test_one_row_per_component()
    print("\nAll tests passed!")
