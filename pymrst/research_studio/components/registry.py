"""M4.1 capability declarations and pre-execution compatibility checks."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import Any, Iterable, Mapping

from pymrst.research_studio.experiments.definition import canonical_value


CAPABILITY_SCHEMA_VERSION = 1


class IncompatibleComponents(ValueError):
    def __init__(self, reasons):
        self.reasons = tuple(str(value) for value in reasons)
        super().__init__("incompatible component selection: " + "; ".join(self.reasons))


@dataclass(frozen=True, slots=True)
class ComponentCapability:
    component_id: str
    kind: str
    owner: str
    phases: tuple[str, ...] = ()
    physics: tuple[str, ...] = ()
    derivatives: tuple[str, ...] = ("none",)
    restart: bool = False
    devices: tuple[str, ...] = ("cpu",)
    precisions: tuple[str, ...] = ("float64",)
    input_schemas: tuple[str, ...] = ()
    output_schemas: tuple[str, ...] = ()
    features: tuple[str, ...] = ()
    requires: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    conflicts: tuple[str, ...] = ()
    version: str = "1"
    schema_version: int = CAPABILITY_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != CAPABILITY_SCHEMA_VERSION:
            raise ValueError("unsupported capability schema")
        if not self.component_id or not self.kind or ":" not in self.owner:
            raise ValueError("component id, kind and module:symbol owner are required")
        for name in ("phases", "physics", "derivatives", "devices", "precisions",
                     "input_schemas", "output_schemas", "features", "conflicts"):
            values = tuple(map(str, getattr(self, name)))
            if len(set(values)) != len(values):
                raise ValueError(f"duplicate {name} capability")
            object.__setattr__(self, name, values)
        requirements = {str(kind): tuple(map(str, values))
                        for kind, values in self.requires.items()}
        object.__setattr__(self, "requires", requirements)

    def to_dict(self):
        return {"schema_version": self.schema_version,
                **{name: getattr(self, name) for name in (
                    "component_id", "kind", "owner", "restart", "version")},
                **{name: list(getattr(self, name)) for name in (
                    "phases", "physics", "derivatives", "devices", "precisions",
                    "input_schemas", "output_schemas", "features", "conflicts")},
                "requires": {key: list(value) for key, value in self.requires.items()}}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)

    def resolve(self):
        module, symbol = self.owner.split(":", 1)
        return getattr(importlib.import_module(module), symbol)


@dataclass(frozen=True, slots=True)
class ComponentRequest:
    selections: Mapping[str, str]
    phases: tuple[str, ...] = ()
    physics: tuple[str, ...] = ()
    derivative: str = "none"
    restart: bool = False
    device: str = "cpu"
    precision: str = "float64"
    required_features: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "selections", canonical_value(self.selections))
        for name in ("phases", "physics", "required_features"):
            object.__setattr__(self, name, tuple(map(str, getattr(self, name))))


class CapabilityRegistry:
    def __init__(self, entries: Iterable[ComponentCapability] = ()):
        self._entries = {}
        for entry in entries:
            self.register(entry)

    def register(self, entry: ComponentCapability):
        if entry.component_id in self._entries:
            raise KeyError(f"component already registered: {entry.component_id}")
        self._entries[entry.component_id] = entry
        return entry

    def get(self, component_id):
        try:
            return self._entries[component_id]
        except KeyError as error:
            raise KeyError(f"unknown component: {component_id}") from error

    def entries(self, *, kind=None):
        values = tuple(self._entries[key] for key in sorted(self._entries))
        return values if kind is None else tuple(value for value in values
                                                  if value.kind == kind)

    def validate(self, request: ComponentRequest):
        selected = {}
        reasons = []
        for slot, component_id in request.selections.items():
            try:
                component = self.get(component_id)
            except KeyError as error:
                reasons.append(str(error))
                continue
            if component.kind != slot:
                reasons.append(f"{component_id} has kind {component.kind}, requested as {slot}")
            selected[slot] = component
        for slot, component in selected.items():
            for phase in request.phases:
                if component.phases and phase not in component.phases:
                    reasons.append(f"{component.component_id} does not support phase {phase}")
            for physics in request.physics:
                if component.physics and physics not in component.physics:
                    reasons.append(f"{component.component_id} does not support physics {physics}")
            if request.derivative not in component.derivatives:
                reasons.append(f"{component.component_id} lacks derivative {request.derivative}")
            if request.restart and not component.restart:
                reasons.append(f"{component.component_id} does not support restart")
            if request.device not in component.devices:
                reasons.append(f"{component.component_id} does not support device {request.device}")
            if request.precision not in component.precisions:
                reasons.append(f"{component.component_id} does not support {request.precision}")
            missing_features = sorted(set(request.required_features)-set(component.features))
            if missing_features:
                reasons.append(f"{component.component_id} lacks features {missing_features}")
            for required_kind, permitted in component.requires.items():
                other = selected.get(required_kind)
                if other is None:
                    reasons.append(f"{component.component_id} requires selection {required_kind}")
                elif permitted and other.component_id not in permitted:
                    reasons.append(f"{component.component_id} requires {required_kind} in {permitted}")
            conflicts = sorted(set(component.conflicts) &
                               {value.component_id for value in selected.values()})
            if conflicts:
                reasons.append(f"{component.component_id} conflicts with {conflicts}")
        if reasons:
            raise IncompatibleComponents(sorted(set(reasons)))
        return selected

    def to_dict(self):
        return {"schema_version": CAPABILITY_SCHEMA_VERSION,
                "components": [value.to_dict() for value in self.entries()]}

    @classmethod
    def from_dict(cls, value):
        if value.get("schema_version") != CAPABILITY_SCHEMA_VERSION:
            raise ValueError("unsupported capability registry schema")
        return cls(ComponentCapability.from_dict(item)
                   for item in value.get("components", ()))


def registered_reference_components():
    """Qualified capabilities of existing MRST-compatible owners."""
    return CapabilityRegistry((
        ComponentCapability(
            "ad-blackoil-reference", "model",
            "pymrst.modules.ad_blackoil.models:GenericBlackOilModel",
            phases=("water", "oil", "gas"), physics=("black-oil",),
            derivatives=("none", "adi", "sparse", "diagonal"), restart=True,
            features=("wells", "fim", "aim", "explicit")),
        ComponentCapability(
            "scipy-direct", "linear_solver",
            "pymrst.modules.ad_core.solvers.concrete_solvers:BackslashSolverAD",
            derivatives=("none", "adi", "sparse", "diagonal"), restart=True),
        ComponentCapability(
            "numpy-scipy-reference", "backend",
            "pymrst.modules.ad_core.backends.sparse.sparse_auto_diff_backend:SparseAutoDiffBackend",
            phases=("water", "oil", "gas"),
            physics=("black-oil", "compositional", "poromechanics"),
            derivatives=("none", "adi", "sparse"), restart=True,
            features=("mrst-parity", "materialized-jacobian")),
    ))


__all__ = ["CAPABILITY_SCHEMA_VERSION", "CapabilityRegistry",
           "ComponentCapability", "ComponentRequest", "IncompatibleComponents",
           "registered_reference_components"]
