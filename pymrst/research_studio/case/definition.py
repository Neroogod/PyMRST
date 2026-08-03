"""Human-readable cases separated from execution profiles."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib
import json
from pathlib import Path
from typing import Any, Mapping

from pymrst.research_studio.experiments.definition import canonical_value, content_digest


CASE_SCHEMA_VERSION = 1
CASE_SECTIONS = ("grid", "properties", "physics", "discretization",
                 "facilities", "schedule", "solver")


class InexpressibleCase(ValueError):
    pass


def _sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CaseArtifact:
    role: str
    path: str
    sha256: str

    @classmethod
    def from_path(cls, role: str, path: str | Path):
        source = Path(path).resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        return cls(role, str(source), _sha256(source))

    def verify(self):
        source = Path(self.path)
        if not source.is_file() or _sha256(source) != self.sha256:
            raise ValueError(f"portable case artifact changed: {self.role} ({source})")

    def to_dict(self):
        return {"role": self.role, "path": self.path, "sha256": self.sha256}


@dataclass(frozen=True, slots=True)
class CaseComponent:
    owner: str
    configuration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if ":" not in self.owner:
            raise ValueError("component owner must use module:symbol syntax")
        object.__setattr__(self, "configuration", canonical_value(self.configuration))

    def to_dict(self):
        return {"owner": self.owner, "configuration": dict(self.configuration)}

    @classmethod
    def from_dict(cls, value):
        if set(value) != {"owner", "configuration"}:
            raise ValueError("invalid case component fields")
        return cls(**value)

    def resolve(self):
        module, name = self.owner.split(":", 1)
        value = getattr(importlib.import_module(module), name)
        if not callable(value):
            raise TypeError(f"case owner is not callable: {self.owner}")
        return value


@dataclass(frozen=True, slots=True)
class ReservoirCase:
    name: str
    builder: CaseComponent
    grid: Mapping[str, Any]
    properties: Mapping[str, Any]
    physics: Mapping[str, Any]
    discretization: Mapping[str, Any]
    facilities: Mapping[str, Any]
    schedule: Mapping[str, Any]
    solver: Mapping[str, Any]
    artifacts: tuple[CaseArtifact, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = CASE_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != CASE_SCHEMA_VERSION:
            raise ValueError(f"unsupported reservoir-case schema {self.schema_version}")
        if not self.name:
            raise ValueError("case name is required")
        for section in CASE_SECTIONS:
            value = canonical_value(getattr(self, section))
            if not isinstance(value, dict):
                raise InexpressibleCase(f"case section {section} must be a mapping")
            object.__setattr__(self, section, value)
        object.__setattr__(self, "artifacts", tuple(self.artifacts))
        object.__setattr__(self, "metadata", canonical_value(self.metadata))
        roles = [item.role for item in self.artifacts]
        if len(set(roles)) != len(roles):
            raise InexpressibleCase("case artifact roles must be unique")

    @property
    def content_hash(self):
        return content_digest(self.to_dict(include_hash=False))

    def to_dict(self, *, include_hash=True):
        value = {"schema_version": self.schema_version, "name": self.name,
                 "builder": self.builder.to_dict(),
                 **{name: dict(getattr(self, name)) for name in CASE_SECTIONS},
                 "artifacts": [item.to_dict() for item in self.artifacts],
                 "metadata": dict(self.metadata)}
        if include_hash:
            value["content_hash"] = self.content_hash
        return value

    @classmethod
    def from_dict(cls, value):
        data = dict(value)
        claimed = data.pop("content_hash", None)
        expected = {"schema_version", "name", "builder", *CASE_SECTIONS,
                    "artifacts", "metadata"}
        if set(data) != expected:
            raise ValueError(f"invalid reservoir-case fields: {sorted(set(data)^expected)}")
        data["builder"] = CaseComponent.from_dict(data["builder"])
        data["artifacts"] = tuple(CaseArtifact(**item) for item in data["artifacts"])
        case = cls(**data)
        if claimed is not None and claimed != case.content_hash:
            raise ValueError("reservoir-case content hash mismatch")
        return case

    def construct(self):
        for artifact in self.artifacts:
            artifact.verify()
        return self.builder.resolve()(**dict(self.builder.configuration))

    @classmethod
    def from_eclipse(cls, prefix: str | Path, *, name="Imported ECLIPSE case"):
        stem = Path(prefix).resolve()
        artifacts = []
        for suffix, role in ((".EGRID", "grid"), (".INIT", "rock"),
                             (".UNRST", "restart")):
            path = stem.with_suffix(suffix)
            if path.exists():
                artifacts.append(CaseArtifact.from_path(role, path))
        if not artifacts or artifacts[0].role != "grid":
            raise FileNotFoundError(f"{stem.with_suffix('.EGRID')} not found")
        return cls(
            name=name,
            builder=CaseComponent(
                "pymrst.research_studio.eclipse.case:read_eclipse_case",
                {"prefix": str(stem)}),
            grid={"kind": "eclipse-egrid", "artifact_role": "grid"},
            properties={"kind": "eclipse-init", "optional": True},
            physics={"kind": "imported", "unit_system": "unknown"},
            discretization={"kind": "source-defined"}, facilities={},
            schedule={"kind": "restart-history", "optional": True}, solver={},
            artifacts=tuple(artifacts), metadata={"source": "ECLIPSE"})


def write_reservoir_case(path: str | Path, case: ReservoirCase):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix+".tmp")
    temporary.write_text(json.dumps(case.to_dict(), indent=2, sort_keys=True,
                                    allow_nan=False)+"\n", encoding="utf-8")
    temporary.replace(target)
    return target


def read_reservoir_case(path: str | Path):
    return ReservoirCase.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


__all__ = ["CASE_SCHEMA_VERSION", "CaseArtifact", "CaseComponent",
           "InexpressibleCase", "ReservoirCase", "read_reservoir_case",
           "write_reservoir_case"]
