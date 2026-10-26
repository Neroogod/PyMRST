"""MRST ``ad-core/statefunctions/StateFunctionDependent.m``.

Mixin for objects whose member functions read state functions - components,
mostly. Each member function records what it reads, so a state function that
calls it (`ComponentPhaseMass` calling `getComponentMass`) can declare exactly
those dependencies.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Union

from pymrst.modules.ad_core.statefunctions.state_function import Dependency


class StateFunctionDependent:
    """`functionDependsOn` / `getFunctionDependencies`."""

    def function_depends_on(self, fn: str, names: Union[str, Sequence[str]],
                            grouping: str = None) -> "StateFunctionDependent":
        """MRST `addPropertyDependence`: no grouping means an internal
        dependency; a grouping (including `'state'`) makes it external."""
        deps = self._function_dependencies()
        if fn not in deps:
            self.clear_function_dependencies(fn)
        names = [names] if isinstance(names, str) else list(names)
        entry = self._function_dependencies()[fn]
        if grouping is None:
            entry["dependencies"] += names
        else:
            entry["externals"] += [Dependency(n, grouping) for n in names]
        return self

    def get_function_dependencies(self, fn: str) -> Dict[str, List[Any]]:
        return self._function_dependencies().get(
            fn, {"dependencies": [], "externals": []})

    def clear_function_dependencies(self, fn: str) -> "StateFunctionDependent":
        self._function_dependencies()[fn] = {"dependencies": [], "externals": []}
        return self

    def _function_dependencies(self) -> Dict[str, Dict[str, List[Any]]]:
        if not hasattr(self, "functionDependencies"):
            self.functionDependencies = {}
        return self.functionDependencies

    functionDependsOn = function_depends_on
    getFunctionDependencies = get_function_dependencies
    clearFunctionDependencies = clear_function_dependencies


__all__ = ["StateFunctionDependent"]


if __name__ == "__main__":
    def test_groupings_make_dependencies_external():
        sfd = StateFunctionDependent()
        sfd.function_depends_on("getComponentMass", ["PoreVolume", "Density"],
                                "PVTPropertyFunctions")
        sfd.function_depends_on("getComponentMass", "s", "state")
        sfd.function_depends_on("getComponentMass", "Other")
        dep = sfd.get_function_dependencies("getComponentMass")
        assert dep["dependencies"] == ["Other"]
        assert [(d.name, d.grouping) for d in dep["externals"]] == [
            ("PoreVolume", "PVTPropertyFunctions"), ("Density", "PVTPropertyFunctions"),
            ("s", "state")]
        print("ok test_groupings_make_dependencies_external")

    def test_an_unknown_function_has_no_dependencies():
        assert StateFunctionDependent().get_function_dependencies("x") == {
            "dependencies": [], "externals": []}
        print("ok test_an_unknown_function_has_no_dependencies")

    test_groupings_make_dependencies_external()
    test_an_unknown_function_has_no_dependencies()
    print("\nAll tests passed!")
