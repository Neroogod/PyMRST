"""Persistent schema for permission-gated simulation run requests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from pymrst.research_studio.runtime.identifiers import CaseId, RunId

from .._serialization import frozen_json_mapping, json_value, require_fields
from ..policies.resource_limits import ExecutionBudget


RUN_REQUEST_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RunRequest:
    request_id: str
    run_id: RunId
    case_id: CaseId
    case_content_hash: str
    execution_profile: str
    model: Mapping[str, Any]
    solver: Mapping[str, Any]
    outputs: tuple[str, ...]
    budget: ExecutionBudget
    variant_id: str | None = None
    random_seed: int = 0
    checkpoint_id: str | None = None
    schema_version: int = RUN_REQUEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUN_REQUEST_SCHEMA_VERSION:
            raise ValueError("unsupported RunRequest schema version")
        for name in ("request_id", "execution_profile"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "run_id", RunId(self.run_id))
        object.__setattr__(self, "case_id", CaseId(self.case_id))
        digest = str(self.case_content_hash).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("case_content_hash must be a lowercase SHA-256 digest")
        object.__setattr__(self, "case_content_hash", digest)
        object.__setattr__(self, "model", frozen_json_mapping(self.model, "$.model"))
        object.__setattr__(self, "solver", frozen_json_mapping(self.solver, "$.solver"))
        outputs = tuple(str(item).strip() for item in self.outputs)
        if not outputs or any(not item for item in outputs):
            raise ValueError("outputs must contain nonempty requested fields")
        if len(set(outputs)) != len(outputs):
            raise ValueError("outputs must be unique")
        object.__setattr__(self, "outputs", outputs)
        seed = int(self.random_seed)
        if seed < 0:
            raise ValueError("random_seed must be nonnegative")
        object.__setattr__(self, "random_seed", seed)
        for name in ("variant_id", "checkpoint_id"):
            value = getattr(self, name)
            if value is not None:
                value = str(value).strip()
                if not value:
                    raise ValueError(f"{name} cannot be empty")
                object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return json_value({
            "schema_version": self.schema_version,
            "request_id": self.request_id, "run_id": str(self.run_id),
            "case_id": str(self.case_id), "case_content_hash": self.case_content_hash,
            "execution_profile": self.execution_profile, "model": self.model,
            "solver": self.solver, "outputs": list(self.outputs),
            "budget": self.budget.to_dict(), "variant_id": self.variant_id,
            "random_seed": self.random_seed, "checkpoint_id": self.checkpoint_id,
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunRequest":
        fields = {
            "schema_version", "request_id", "run_id", "case_id",
            "case_content_hash", "execution_profile", "model", "solver",
            "outputs", "budget", "variant_id", "random_seed", "checkpoint_id",
        }
        data = require_fields(value, required=fields, name="RunRequest")
        data["budget"] = ExecutionBudget.from_dict(data["budget"])
        data["outputs"] = tuple(data["outputs"])
        return cls(**data)


__all__ = ["RUN_REQUEST_SCHEMA_VERSION", "RunRequest"]
