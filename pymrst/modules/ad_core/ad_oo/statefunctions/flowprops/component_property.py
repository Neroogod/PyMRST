"""MRST ``ad-core/statefunctions/flowprops/ComponentProperty.m``.

Mixin for state functions computed per component by asking each of the
model's `Components` to evaluate one named member function. It collects those
functions' declared dependencies so the property depends on exactly what its
components read.
"""

from __future__ import annotations

from typing import Any, List


class ComponentProperty:
    """Declares the dependencies of `componentFunctionName` on each component."""

    def init_component_property(self, model: Any, name: str) -> None:
        self.componentFunctionName = name
        self.includeShrinkageFactors = None
        dependencies: List[Any] = []
        externals: List[Any] = []
        for component in model.Components:
            found = component.get_function_dependencies(name)
            dependencies += list(found.get("dependencies", []))
            externals += list(found.get("externals", []))
        for dependency in sorted(set(dependencies)):
            self.depends_on(dependency)
        seen = set()
        for external in externals:
            if external.name not in seen:
                seen.add(external.name)
                self.depends_on(external.name, external.grouping)
        if self.includeShrinkageFactors is None:
            from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
                ThreePhaseBlackOilModel,
            )
            self.includeShrinkageFactors = isinstance(model, ThreePhaseBlackOilModel)


__all__ = ["ComponentProperty"]


if __name__ == "__main__":
    from types import SimpleNamespace

    from pymrst.modules.ad_core.statefunctions.state_function import (
        Dependency,
        StateFunction,
    )

    class _Prop(StateFunction, ComponentProperty):
        pass

    def test_dependencies_are_collected_from_every_component():
        component = SimpleNamespace(get_function_dependencies=lambda name: {
            "dependencies": ["Density"],
            "externals": [Dependency("Viscosity", "PVTPropertyFunctions")]})
        model = SimpleNamespace(Components=[component, component])
        prop = _Prop()
        prop.init_component_property(model, "getComponentMass")
        names = [(d.name, d.grouping) for d in prop.dependencies]
        assert ("Density", "self") in names
        assert names.count(("Viscosity", "PVTPropertyFunctions")) == 1
        assert prop.includeShrinkageFactors is False
        print("ok test_dependencies_are_collected_from_every_component")

    test_dependencies_are_collected_from_every_component()
    print("\nAll tests passed!")
