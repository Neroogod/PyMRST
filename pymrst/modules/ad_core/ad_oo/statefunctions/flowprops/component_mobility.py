"""MRST ``ad-core/statefunctions/flowprops/ComponentMobility.m``.

Mobility of each component in each phase (mass, not volume), from every
component's `getComponentMobility`.
"""

from __future__ import annotations

from typing import Any, List

from pymrst.modules.ad_core.ad_oo.statefunctions.flowprops.component_property import (
    ComponentProperty,
)
from pymrst.modules.ad_core.statefunctions.state_function import StateFunction


class ComponentMobility(StateFunction, ComponentProperty):
    """`lambda_{i,alpha}`, `ncomp x nphase`."""

    def __init__(self, model: Any, regions: Any = None):
        StateFunction.__init__(self, model, label="\\lambda_{i, \\alpha}",
                               regions=regions)
        self.AutoDiffBackend = model.AutoDiffBackend
        self.init_component_property(model, "getComponentMobility")

    def evaluate_on_domain(self, model: Any, state: Any) -> List[List[Any]]:
        return [list(component.get_component_mobility(model, state))
                for component in model.Components]

    evaluateOnDomain = evaluate_on_domain


__all__ = ["ComponentMobility"]


if __name__ == "__main__":
    from types import SimpleNamespace

    def test_one_row_per_component():
        component = SimpleNamespace(
            get_function_dependencies=lambda name: {},
            get_component_mobility=lambda model, state: [2.0, None])
        model = SimpleNamespace(Components=[component], AutoDiffBackend=None)
        assert ComponentMobility(model).evaluate_on_domain(model, {}) == [[2.0, None]]
        print("ok test_one_row_per_component")

    test_one_row_per_component()
    print("\nAll tests passed!")
