"""Persistent schema for agent-visible simulation run results."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from pymrst.research_studio.runtime.identifiers import RunId

from .._serialization import frozen_json_mapping, json_value, require_fields
from .evidence import EvidenceReference


RUN_RESULT_SCHEMA_VERSION = 1


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class RunFailure:
    error_type: str
    message: str
    retryable: bool = False
    failure_bundle_uri: str | None = None

    def __post_init__(self) -> None:
        for name in ("error_type", "message"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        if self.failure_bundle_uri is not None:
            uri = str(self.failure_bundle_uri).strip()
            if "://" not in uri:
                raise ValueError("failure_bundle_uri must be an application URI")
            object.__setattr__(self, "failure_bundle_uri", uri)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error_type": self.error_type, "message": self.message,
            "retryable": self.retryable,
            "failure_bundle_uri": self.failure_bundle_uri,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunFailure":
        return cls(**require_fields(
            value,
            required={"error_type", "message", "retryable", "failure_bundle_uri"},
            name="RunFailure",
        ))


@dataclass(frozen=True, slots=True)
class RunResult:
    request_id: str
    run_id: RunId
    status: RunStatus
    result_store_uri: str | None = None
    evidence: tuple[EvidenceReference, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    failure: RunFailure | None = None
    schema_version: int = RUN_RESULT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RUN_RESULT_SCHEMA_VERSION:
            raise ValueError("unsupported RunResult schema version")
        request_id = str(self.request_id).strip()
        if not request_id:
            raise ValueError("request_id cannot be empty")
        object.__setattr__(self, "request_id", request_id)
        object.__setattr__(self, "run_id", RunId(self.run_id))
        object.__setattr__(self, "status", RunStatus(self.status))
        if self.result_store_uri is not None:
            uri = str(self.result_store_uri).strip()
            if "://" not in uri:
                raise ValueError("result_store_uri must be an application URI")
            object.__setattr__(self, "result_store_uri", uri)
        evidence = tuple(
            item if isinstance(item, EvidenceReference)
            else EvidenceReference.from_dict(item) for item in self.evidence
        )
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(self, "metrics", frozen_json_mapping(self.metrics, "$.metrics"))
        if self.status is RunStatus.COMPLETED and self.result_store_uri is None:
            raise ValueError("completed runs require a result_store_uri")
        if self.status is RunStatus.FAILED and self.failure is None:
            raise ValueError("failed runs require failure details")
        if self.status is not RunStatus.FAILED and self.failure is not None:
            raise ValueError("only failed runs may contain failure details")

    def to_dict(self) -> dict[str, Any]:
        return json_value({
            "schema_version": self.schema_version,
            "request_id": self.request_id, "run_id": str(self.run_id),
            "status": self.status.value, "result_store_uri": self.result_store_uri,
            "evidence": [item.to_dict() for item in self.evidence],
            "metrics": self.metrics,
            "failure": None if self.failure is None else self.failure.to_dict(),
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunResult":
        fields = {
            "schema_version", "request_id", "run_id", "status",
            "result_store_uri", "evidence", "metrics", "failure",
        }
        data = require_fields(value, required=fields, name="RunResult")
        data["evidence"] = tuple(
            EvidenceReference.from_dict(item) for item in data["evidence"]
        )
        if data["failure"] is not None:
            data["failure"] = RunFailure.from_dict(data["failure"])
        return cls(**data)


__all__ = ["RUN_RESULT_SCHEMA_VERSION", "RunFailure", "RunResult", "RunStatus"]
