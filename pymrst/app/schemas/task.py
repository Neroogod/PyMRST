"""Durable background-task contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from ._common import json_value, nonempty


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class TaskRecord:
    task_id: str
    project_id: str
    kind: str
    status: TaskStatus
    progress: float
    message: str
    request: Mapping[str, Any] = field(default_factory=dict)
    result: Mapping[str, Any] | None = None
    error: Mapping[str, Any] | None = None
    created_at: str = ""
    updated_at: str = ""
    cancellation_requested: bool = False

    def __post_init__(self) -> None:
        for name in ("task_id", "project_id", "kind", "message", "created_at", "updated_at"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        object.__setattr__(self, "status", TaskStatus(self.status))
        progress = float(self.progress)
        if not 0 <= progress <= 1:
            raise ValueError("task progress must lie in [0, 1]")
        if self.status is TaskStatus.COMPLETED and progress != 1:
            raise ValueError("completed tasks require progress 1")
        if self.status is TaskStatus.FAILED and self.error is None:
            raise ValueError("failed tasks require structured error details")
        object.__setattr__(self, "progress", progress)
        object.__setattr__(self, "request", json_value(self.request, "$.request"))
        object.__setattr__(self, "result", None if self.result is None else json_value(self.result, "$.result"))
        object.__setattr__(self, "error", None if self.error is None else json_value(self.error, "$.error"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id, "project_id": self.project_id,
            "kind": self.kind, "status": self.status.value,
            "progress": self.progress, "message": self.message,
            "request": self.request, "result": self.result, "error": self.error,
            "created_at": self.created_at, "updated_at": self.updated_at,
            "cancellation_requested": self.cancellation_requested,
        }


__all__ = ["TaskRecord", "TaskStatus"]
