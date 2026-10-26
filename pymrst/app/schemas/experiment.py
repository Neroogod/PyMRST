"""Application contracts for governed experiments, ensembles, and qualification."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from ._common import json_value, nonempty


class ExperimentSourceKind(StrEnum):
    MODEL_REVISION = "model_revision"
    RUN_CONFIGURATION = "run_configuration"


class ExecutionTargetKind(StrEnum):
    LOCAL = "local"
    SLURM = "slurm"
    SSH = "ssh"


@dataclass(frozen=True, slots=True)
class ExperimentRequest:
    name: str
    hypothesis: str
    description: str
    source_kind: ExperimentSourceKind
    source_id: str
    matrix: Mapping[str, Any]
    member_runner: str = "manufactured-qualification"
    outputs: tuple[str, ...] = ("pressure",)
    random_seed: int = 0
    tags: tuple[str, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "name", nonempty(self.name, "experiment name"))
        object.__setattr__(self, "hypothesis", nonempty(self.hypothesis, "hypothesis"))
        object.__setattr__(self, "description", nonempty(self.description, "description"))
        object.__setattr__(self, "source_kind", ExperimentSourceKind(self.source_kind))
        object.__setattr__(self, "source_id", nonempty(self.source_id, "source id"))
        object.__setattr__(self, "member_runner", nonempty(self.member_runner, "member runner"))
        object.__setattr__(self, "matrix", json_value(self.matrix))
        outputs = tuple(nonempty(item, "output field") for item in self.outputs)
        if not outputs or len(outputs) != len(set(outputs)):
            raise ValueError("output fields must be nonempty and unique")
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "tags", tuple(sorted(set(map(str, self.tags)))))
        if not 0 <= int(self.random_seed) < 2**63:
            raise ValueError("experiment seed must lie in [0, 2^63)")
        object.__setattr__(self, "random_seed", int(self.random_seed))

    @classmethod
    def from_dict(cls, value):
        return cls(
            name=value["name"], hypothesis=value["hypothesis"],
            description=value["description"], source_kind=value["source_kind"],
            source_id=value["source_id"], matrix=value["matrix"],
            member_runner=value.get("member_runner", "manufactured-qualification"),
            outputs=tuple(value.get("outputs", ("pressure",))),
            random_seed=value.get("random_seed", 0), tags=tuple(value.get("tags", ())),
        )


@dataclass(frozen=True, slots=True)
class ExecutionTarget:
    kind: ExecutionTargetKind = ExecutionTargetKind.LOCAL
    name: str = "Local workstation"
    max_workers: int = 1
    memory_per_worker_bytes: int = 512*1024*1024
    configuration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "kind", ExecutionTargetKind(self.kind))
        object.__setattr__(self, "name", nonempty(self.name, "execution target name"))
        workers, memory = int(self.max_workers), int(self.memory_per_worker_bytes)
        if not 1 <= workers <= 256:
            raise ValueError("max_workers must lie in [1,256]")
        if not 64*1024*1024 <= memory <= 1024**4:
            raise ValueError("memory_per_worker_bytes must lie in [64 MiB, 1 TiB]")
        configuration = json_value(self.configuration)
        forbidden = {key.lower() for key in configuration} & {
            "password", "token", "secret", "private_key", "api_key",
        }
        if forbidden:
            raise ValueError("execution profiles cannot persist credentials")
        object.__setattr__(self, "max_workers", workers)
        object.__setattr__(self, "memory_per_worker_bytes", memory)
        object.__setattr__(self, "configuration", configuration)

    def to_dict(self):
        return {"kind": self.kind.value, "name": self.name,
                "max_workers": self.max_workers,
                "memory_per_worker_bytes": self.memory_per_worker_bytes,
                "configuration": dict(self.configuration)}

    @classmethod
    def from_dict(cls, value):
        return cls(**value)


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    experiment_id: str; project_id: str; source_kind: str; source_id: str
    name: str; content_hash: str; member_count: int
    definition: Mapping[str, Any]; matrix: Mapping[str, Any]
    coordinates: tuple[Mapping[str, Any], ...]; created_at: str

    def to_dict(self):
        return {"experiment_id": self.experiment_id, "project_id": self.project_id,
                "source_kind": self.source_kind, "source_id": self.source_id,
                "name": self.name, "content_hash": self.content_hash,
                "member_count": self.member_count, "definition": dict(self.definition),
                "matrix": dict(self.matrix),
                "coordinates": [dict(item) for item in self.coordinates],
                "created_at": self.created_at}


@dataclass(frozen=True, slots=True)
class ExperimentExecution:
    execution_id: str; experiment_id: str; project_id: str; target_kind: str
    configuration_id: str; run_id: str; status: str; target: Mapping[str, Any]
    relative_path: str; summary: Mapping[str, Any]; created_at: str; updated_at: str

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True)
class QualificationRecord:
    qualification_id: str; experiment_id: str; execution_id: str
    reference_kind: str; reference_id: str; status: str
    tolerances: Mapping[str, Any]; metrics: Mapping[str, Any]
    evidence: Mapping[str, Any]; created_at: str

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True)
class BranchRequest:
    name: str
    parent_member_id: str
    parent_time_index: int = -1
    parameters: Mapping[str, Any] = field(default_factory=dict)
    components: Mapping[str, str] = field(default_factory=dict)
    controls: Mapping[str, Any] = field(default_factory=dict)
    properties: Mapping[str, Any] = field(default_factory=dict)
    required_fields: tuple[str, ...] = ("pressure",)
    runner_kind: str = "manufactured-branch"

    def __post_init__(self):
        object.__setattr__(self, "name", nonempty(self.name, "branch name"))
        object.__setattr__(self, "parent_member_id", nonempty(
            self.parent_member_id, "parent member id"))
        object.__setattr__(self, "parent_time_index", int(self.parent_time_index))
        for name in ("parameters", "components", "controls", "properties"):
            object.__setattr__(self, name, json_value(getattr(self, name)))
        object.__setattr__(self, "required_fields", tuple(
            nonempty(item, "required branch field") for item in self.required_fields))
        object.__setattr__(self, "runner_kind", nonempty(self.runner_kind, "branch runner"))

    @classmethod
    def from_dict(cls, value):
        return cls(
            name=value["name"], parent_member_id=value["parent_member_id"],
            parent_time_index=value.get("parent_time_index", -1),
            parameters=value.get("parameters", {}),
            components=value.get("components", {}), controls=value.get("controls", {}),
            properties=value.get("properties", {}),
            required_fields=tuple(value.get("required_fields", ("pressure",))),
            runner_kind=value.get("runner_kind", "manufactured-branch"),
        )


@dataclass(frozen=True, slots=True)
class BranchRecord:
    branch_id: str; experiment_id: str; execution_id: str; project_id: str
    parent_member_id: str; configuration_id: str; run_id: str; status: str
    definition: Mapping[str, Any]; relative_path: str; summary: Mapping[str, Any]
    created_at: str; updated_at: str

    def to_dict(self):
        return {name: getattr(self, name) for name in self.__slots__}


__all__ = [
    "ExecutionTarget", "ExecutionTargetKind", "ExperimentExecution",
    "ExperimentRecord", "ExperimentRequest", "ExperimentSourceKind",
    "QualificationRecord",
    "BranchRecord", "BranchRequest",
]
