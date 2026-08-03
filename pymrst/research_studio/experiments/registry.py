"""Collision-safe, persistent registry of immutable experiment definitions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from pymrst.research_studio.runtime.identifiers import ExperimentId

from .definition import ExperimentDefinition


EXPERIMENT_REGISTRY_SCHEMA_VERSION = 1


class ExperimentIdentityCollision(RuntimeError):
    pass


class ExperimentRegistry:
    def __init__(self, definitions: Iterable[ExperimentDefinition] = ()):
        self._definitions: dict[ExperimentId, ExperimentDefinition] = {}
        self._names: dict[str, ExperimentId] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: ExperimentDefinition) -> ExperimentId:
        if not isinstance(definition, ExperimentDefinition):
            raise TypeError("registry values must be ExperimentDefinition instances")
        identity = definition.experiment_id
        existing = self._definitions.get(identity)
        if existing is not None:
            if existing.to_dict() != definition.to_dict():
                raise ExperimentIdentityCollision(
                    f"experiment digest collision for {identity}")
            return identity
        named = self._names.get(definition.name)
        if named is not None and named != identity:
            raise ValueError(
                f"experiment name {definition.name!r} already identifies {named}")
        self._definitions[identity] = definition
        self._names[definition.name] = identity
        return identity

    @property
    def experiment_ids(self) -> tuple[ExperimentId, ...]:
        return tuple(sorted(self._definitions, key=str))

    def get(self, identity: ExperimentId | str) -> ExperimentDefinition:
        return self._definitions[ExperimentId(identity)]

    def by_name(self, name: str) -> ExperimentDefinition:
        return self.get(self._names[name])

    def to_dict(self):
        return {
            "schema_version": EXPERIMENT_REGISTRY_SCHEMA_VERSION,
            "definitions": [self._definitions[key].to_dict()
                            for key in self.experiment_ids],
        }

    @classmethod
    def from_dict(cls, value):
        if set(value) != {"schema_version", "definitions"}:
            raise ValueError("invalid ExperimentRegistry fields")
        if value["schema_version"] != EXPERIMENT_REGISTRY_SCHEMA_VERSION:
            raise ValueError("unsupported ExperimentRegistry schema")
        return cls(ExperimentDefinition.from_dict(item) for item in value["definitions"])

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(target)

    @classmethod
    def load(cls, path: str | Path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = ["EXPERIMENT_REGISTRY_SCHEMA_VERSION", "ExperimentIdentityCollision",
           "ExperimentRegistry"]
