"""Versioned, JSON-safe evidence records with strict load validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
import json
import math
from typing import Any, Mapping

from pymrst.research_studio.runtime.identifiers import (
    CaseId, ExperimentId, MemberId, RunId,
)

EVIDENCE_SCHEMA_VERSION = 1


def _json_value(value: Any, path: str = "$") -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"non-finite number at {path}")
        return value
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item, f"{path}[{index}]") for index, item in enumerate(value)]
    raise TypeError(f"unsupported persistent value {type(value).__name__} at {path}")


@dataclass(frozen=True, slots=True)
class RunIdentity:
    run_id: RunId
    case_id: CaseId
    experiment_id: ExperimentId | None = None
    member_id: MemberId | None = None


@dataclass(slots=True)
class SimulationEvidence:
    """Persistent evidence envelope; it never contains model/solver objects."""

    run_identity: RunIdentity
    schema_version: int = EVIDENCE_SCHEMA_VERSION
    manifest: dict[str, Any] = field(default_factory=dict)
    input_lineage: dict[str, Any] = field(default_factory=dict)
    timestep_reports: list[dict[str, Any]] = field(default_factory=list)
    convergence: dict[str, Any] = field(default_factory=dict)
    conservation: dict[str, Any] = field(default_factory=dict)
    invariants: list[dict[str, Any]] = field(default_factory=list)
    deck_coverage: dict[str, Any] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[dict[str, Any]] = field(default_factory=list)
    timings: dict[str, Any] = field(default_factory=dict)
    comparisons: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return _json_value(asdict(self))

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "SimulationEvidence":
        data = dict(value)
        version = data.get("schema_version")
        if version != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported SimulationEvidence schema version {version!r}")
        identity = data.get("run_identity")
        if not isinstance(identity, Mapping):
            raise ValueError("run_identity must be an object")
        allowed = {field.name for field in cls.__dataclass_fields__.values()}
        unknown = set(data) - allowed
        if unknown:
            raise ValueError(f"unknown SimulationEvidence fields: {sorted(unknown)}")
        # Validate the wire representation before replacing the identity
        # mapping with its strongly typed in-memory form.
        _json_value(data)
        data["run_identity"] = RunIdentity(
            run_id=RunId(identity["run_id"]),
            case_id=CaseId(identity["case_id"]),
            experiment_id=(ExperimentId(identity["experiment_id"])
                           if identity.get("experiment_id") is not None else None),
            member_id=(MemberId(identity["member_id"])
                       if identity.get("member_id") is not None else None),
        )
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> "SimulationEvidence":
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("SimulationEvidence JSON root must be an object")
        return cls.from_dict(parsed)


__all__ = ["EVIDENCE_SCHEMA_VERSION", "RunIdentity", "SimulationEvidence"]
