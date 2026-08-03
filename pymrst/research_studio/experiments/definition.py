"""Versioned, canonical experiment definitions for reproducible studies."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

from pymrst.research_studio.runtime.identifiers import CaseId, ExperimentId


EXPERIMENT_DEFINITION_SCHEMA_VERSION = 1


def canonical_value(value: Any) -> Any:
    """Return the strict JSON value used by experiment identity v1."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("experiment values must be finite")
        return 0.0 if value == 0.0 else value
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError("experiment mapping keys must be nonempty strings")
            normalized[key] = canonical_value(item)
        return {key: normalized[key] for key in sorted(normalized)}
    if isinstance(value, (list, tuple)):
        return [canonical_value(item) for item in value]
    raise TypeError(
        f"unsupported experiment value {type(value).__name__}; use JSON-safe data")


def canonical_json(value: Any) -> str:
    return json.dumps(canonical_value(value), sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def content_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _validate_sha256(value: str, owner: str) -> str:
    digest = str(value).lower()
    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError(f"{owner} must be a lowercase SHA-256 digest")
    return digest


@dataclass(frozen=True, slots=True)
class DatasetReference:
    name: str
    sha256: str
    size_bytes: int
    source: str = ""
    media_type: str = "application/octet-stream"

    def __post_init__(self):
        if not self.name or self.size_bytes < 0 or not self.media_type:
            raise ValueError("dataset name, nonnegative size, and media type are required")
        object.__setattr__(self, "sha256", _validate_sha256(self.sha256, "dataset sha256"))

    @classmethod
    def from_path(cls, name: str, path: str | Path, *, media_type=None):
        source = Path(path).resolve()
        digest = hashlib.sha256()
        with source.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return cls(name=name, sha256=digest.hexdigest(), size_bytes=source.stat().st_size,
                   source=str(source), media_type=media_type or "application/octet-stream")

    def identity_dict(self):
        return {"name": self.name, "sha256": self.sha256,
                "size_bytes": self.size_bytes, "media_type": self.media_type}

    def verify(self) -> None:
        if not self.source:
            raise ValueError(f"dataset {self.name!r} has no verifiable local source")
        source = Path(self.source)
        if not source.is_file():
            raise FileNotFoundError(f"dataset source is missing: {source}")
        current = DatasetReference.from_path(
            self.name, source, media_type=self.media_type)
        if (current.sha256 != self.sha256 or current.size_bytes != self.size_bytes):
            raise ValueError(f"dataset content changed: {source}")

    def to_dict(self):
        return {**self.identity_dict(), "source": self.source}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True, slots=True)
class OwnerReference:
    owner: str
    configuration: Mapping[str, Any] = field(default_factory=dict)
    fingerprint: str = ""

    def __post_init__(self):
        if not self.owner or any(character.isspace() for character in self.owner):
            raise ValueError("owner must be a nonempty import-style identifier")
        object.__setattr__(self, "configuration", canonical_value(self.configuration))
        if self.fingerprint:
            object.__setattr__(self, "fingerprint",
                               _validate_sha256(self.fingerprint, "owner fingerprint"))

    def to_dict(self):
        return {"owner": self.owner, "configuration": dict(self.configuration),
                "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True, slots=True)
class CaseReference:
    case_id: CaseId
    constructor: OwnerReference
    content_hash: str

    def __post_init__(self):
        object.__setattr__(self, "case_id", CaseId(self.case_id))
        object.__setattr__(self, "content_hash",
                           _validate_sha256(self.content_hash, "case content hash"))

    def to_dict(self):
        return {"case_id": str(self.case_id), "constructor": self.constructor.to_dict(),
                "content_hash": self.content_hash}

    @classmethod
    def from_dict(cls, value):
        return cls(case_id=CaseId(value["case_id"]),
                   constructor=OwnerReference.from_dict(value["constructor"]),
                   content_hash=value["content_hash"])


@dataclass(frozen=True, slots=True)
class ExecutionProfile:
    name: str
    configuration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name:
            raise ValueError("execution profile name is required")
        object.__setattr__(self, "configuration", canonical_value(self.configuration))

    def to_dict(self):
        return {"name": self.name, "configuration": dict(self.configuration)}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ValidationRequirement:
    name: str
    policy: Mapping[str, Any]

    def __post_init__(self):
        if not self.name or not self.policy:
            raise ValueError("validation requirement name and policy are required")
        object.__setattr__(self, "policy", canonical_value(self.policy))

    def to_dict(self):
        return {"name": self.name, "policy": dict(self.policy)}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True, slots=True)
class OutputRequest:
    name: str
    fields: tuple[str, ...]
    cadence: str = "report_step"

    def __post_init__(self):
        if not self.name or not self.fields or not self.cadence:
            raise ValueError("output name, fields, and cadence are required")
        fields = tuple(str(value) for value in self.fields)
        if len(set(fields)) != len(fields):
            raise ValueError("output fields must be unique")
        object.__setattr__(self, "fields", fields)

    def to_dict(self):
        return {"name": self.name, "fields": list(self.fields), "cadence": self.cadence}

    @classmethod
    def from_dict(cls, value):
        return cls(name=value["name"], fields=tuple(value["fields"]),
                   cadence=value.get("cadence", "report_step"))


@dataclass(frozen=True, slots=True)
class ExperimentDefinition:
    name: str
    hypothesis: str
    description: str
    base_case: CaseReference
    model: OwnerReference
    schedule: OwnerReference
    variants: tuple["VariantDefinition", ...]
    datasets: tuple[DatasetReference, ...]
    execution_profile: ExecutionProfile
    validation: tuple[ValidationRequirement, ...]
    outputs: tuple[OutputRequest, ...]
    publication_artifacts: tuple[str, ...] = ()
    random_seed: int = 0
    tags: tuple[str, ...] = ()
    schema_version: int = EXPERIMENT_DEFINITION_SCHEMA_VERSION

    def __post_init__(self):
        from .variants import VariantDefinition
        if self.schema_version != EXPERIMENT_DEFINITION_SCHEMA_VERSION:
            raise ValueError("unsupported ExperimentDefinition schema version")
        if not self.name or not self.hypothesis or not self.description:
            raise ValueError("experiment name, hypothesis, and description are required")
        if self.random_seed < 0:
            raise ValueError("random_seed must be nonnegative")
        for owner, values in (("variant", self.variants), ("dataset", self.datasets),
                              ("validation", self.validation), ("output", self.outputs)):
            names = [value.name for value in values]
            if len(set(names)) != len(names):
                raise ValueError(f"{owner} names must be unique")
        if not all(isinstance(item, VariantDefinition) for item in self.variants):
            raise TypeError("variants must contain VariantDefinition values")
        tags = tuple(sorted(set(str(value) for value in self.tags)))
        object.__setattr__(self, "tags", tags)
        object.__setattr__(self, "publication_artifacts",
                           tuple(str(value) for value in self.publication_artifacts))

    def execution_dict(self):
        return {
            "schema_version": self.schema_version,
            "base_case": self.base_case.to_dict(), "model": self.model.to_dict(),
            "schedule": self.schedule.to_dict(),
            "variants": [value.to_dict() for value in self.variants],
            "datasets": [value.identity_dict() for value in self.datasets],
            "execution_profile": self.execution_profile.to_dict(),
            "random_seed": self.random_seed,
        }

    def to_dict(self):
        return {
            **self.execution_dict(), "name": self.name, "hypothesis": self.hypothesis,
            "description": self.description,
            "datasets": [value.to_dict() for value in self.datasets],
            "validation": [value.to_dict() for value in self.validation],
            "outputs": [value.to_dict() for value in self.outputs],
            "publication_artifacts": list(self.publication_artifacts),
            "tags": list(self.tags),
        }

    @property
    def execution_fingerprint(self) -> str:
        return content_digest(self.execution_dict())

    @property
    def experiment_id(self) -> ExperimentId:
        payload = self.to_dict()
        # Source locations are provenance and must not make machine identity differ.
        payload["datasets"] = [value.identity_dict() for value in self.datasets]
        return ExperimentId(f"experiment_{content_digest(payload)}")

    def to_json(self, *, indent=2):
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True,
                          ensure_ascii=False, allow_nan=False)

    def verify_datasets(self) -> None:
        for dataset in self.datasets:
            dataset.verify()

    @classmethod
    def from_dict(cls, value):
        from .variants import VariantDefinition
        allowed = set(cls.__dataclass_fields__)
        if set(value) != allowed:
            raise ValueError(f"invalid ExperimentDefinition fields: {sorted(set(value)^allowed)}")
        return cls(
            name=value["name"], hypothesis=value["hypothesis"],
            description=value["description"],
            base_case=CaseReference.from_dict(value["base_case"]),
            model=OwnerReference.from_dict(value["model"]),
            schedule=OwnerReference.from_dict(value["schedule"]),
            variants=tuple(VariantDefinition.from_dict(item) for item in value["variants"]),
            datasets=tuple(DatasetReference.from_dict(item) for item in value["datasets"]),
            execution_profile=ExecutionProfile.from_dict(value["execution_profile"]),
            validation=tuple(ValidationRequirement.from_dict(item)
                             for item in value["validation"]),
            outputs=tuple(OutputRequest.from_dict(item) for item in value["outputs"]),
            publication_artifacts=tuple(value["publication_artifacts"]),
            random_seed=int(value["random_seed"]), tags=tuple(value["tags"]),
            schema_version=int(value["schema_version"]),
        )

    @classmethod
    def from_json(cls, value: str):
        return cls.from_dict(json.loads(value))


__all__ = ["EXPERIMENT_DEFINITION_SCHEMA_VERSION", "CaseReference",
           "DatasetReference", "ExecutionProfile", "ExperimentDefinition",
           "OutputRequest", "OwnerReference", "ValidationRequirement",
           "canonical_json", "canonical_value", "content_digest"]
