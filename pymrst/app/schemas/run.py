"""Versioned contracts for isolated application simulation runs."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from ._common import json_value, nonempty


class RunStatus(StrEnum):
    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    CANCELLING = "cancelling"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class RunConfiguration:
    configuration_id: str
    project_id: str
    model_id: str
    name: str
    runner_kind: str
    configuration_hash: str
    payload: Mapping[str, Any]
    created_at: str

    def __post_init__(self) -> None:
        for name in ("configuration_id", "project_id", "name", "runner_kind", "created_at"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        object.__setattr__(self, "model_id", str(self.model_id))
        digest = str(self.configuration_hash).lower()
        if len(digest) != 64 or any(item not in "0123456789abcdef" for item in digest):
            raise ValueError("configuration_hash must be SHA-256")
        object.__setattr__(self, "configuration_hash", digest)
        object.__setattr__(self, "payload", json_value(dict(self.payload)))

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    project_id: str
    configuration_id: str
    task_id: str
    attempt_number: int
    parent_run_id: str
    resume_checkpoint_id: str
    status: RunStatus
    run_relative_path: str
    process_id: int | None
    started_at: str
    finished_at: str
    created_at: str

    def __post_init__(self) -> None:
        for name in ("run_id", "project_id", "configuration_id", "task_id", "run_relative_path", "created_at"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        object.__setattr__(self, "status", RunStatus(self.status))
        if int(self.attempt_number) < 1:
            raise ValueError("run attempt number must be positive")
        object.__setattr__(self, "attempt_number", int(self.attempt_number))
        if self.process_id is not None:
            object.__setattr__(self, "process_id", int(self.process_id))

    def to_dict(self) -> dict[str, Any]:
        value = {name: getattr(self, name) for name in self.__slots__}
        value["status"] = self.status.value
        return value


@dataclass(frozen=True, slots=True)
class RunEvent:
    run_id: str
    sequence: int
    event_type: str
    elapsed_seconds: float
    payload: Mapping[str, Any]
    created_at: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_id", nonempty(self.run_id, "run_id"))
        object.__setattr__(self, "event_type", nonempty(self.event_type, "event_type"))
        if int(self.sequence) < 0 or float(self.elapsed_seconds) < 0:
            raise ValueError("run event sequence and elapsed time cannot be negative")
        object.__setattr__(self, "sequence", int(self.sequence))
        object.__setattr__(self, "elapsed_seconds", float(self.elapsed_seconds))
        object.__setattr__(self, "payload", json_value(dict(self.payload)))

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True)
class RunCheckpoint:
    checkpoint_id: str
    run_id: str
    sequence: int
    relative_path: str
    content_hash: str
    metadata: Mapping[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}


__all__ = ["RunCheckpoint", "RunConfiguration", "RunEvent", "RunRecord", "RunStatus"]
