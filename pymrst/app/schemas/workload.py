"""Contracts for installation-wide workload and performance qualification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ._common import json_value, nonempty


class WorkloadTier(StrEnum):
    SMOKE = "smoke"
    ENGINEERING = "engineering"
    FIELD_SCALE = "field_scale"
    ENDURANCE = "endurance"


class WorkloadStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class WorkloadBudget:
    maximum_wall_seconds: float
    maximum_cpu_seconds: float
    maximum_peak_rss_bytes: int
    maximum_disk_bytes: int
    maximum_wall_regression_ratio: float = 1.25
    maximum_memory_regression_ratio: float = 1.25
    maximum_wall_relative_mad: float = 0.15

    def __post_init__(self):
        for name in ("maximum_wall_seconds", "maximum_cpu_seconds"):
            value = float(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in ("maximum_peak_rss_bytes", "maximum_disk_bytes"):
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        for name in ("maximum_wall_regression_ratio", "maximum_memory_regression_ratio"):
            value = float(getattr(self, name))
            if value < 1:
                raise ValueError(f"{name} cannot be below one")
            object.__setattr__(self, name, value)
        relative_mad = float(self.maximum_wall_relative_mad)
        if not 0 <= relative_mad <= 1:
            raise ValueError("maximum_wall_relative_mad must lie in [0,1]")
        object.__setattr__(self, "maximum_wall_relative_mad", relative_mad)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WorkloadBudget":
        return cls(**dict(value))

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


@dataclass(frozen=True, slots=True)
class WorkloadDefinition:
    workload_id: str
    title: str
    description: str
    tiers: tuple[WorkloadTier, ...]
    default_budgets: dict[str, WorkloadBudget]
    available: bool = True
    unavailable_reason: str = ""
    requirements: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("workload_id", "title", "description"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        object.__setattr__(self, "tiers", tuple(WorkloadTier(item) for item in self.tiers))
        object.__setattr__(self, "default_budgets", {
            WorkloadTier(key).value: value if isinstance(value, WorkloadBudget)
            else WorkloadBudget.from_dict(value)
            for key, value in self.default_budgets.items()
        })
        if set(self.default_budgets) != {item.value for item in self.tiers}:
            raise ValueError("every workload tier requires one default budget")
        object.__setattr__(self, "available", bool(self.available))
        object.__setattr__(self, "unavailable_reason", str(self.unavailable_reason).strip())
        object.__setattr__(self, "requirements", tuple(
            nonempty(item, "requirement") for item in self.requirements
        ))
        if not self.available and not self.unavailable_reason:
            raise ValueError("an unavailable workload requires a reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id, "title": self.title,
            "description": self.description,
            "tiers": [item.value for item in self.tiers],
            "default_budgets": {
                key: value.to_dict() for key, value in self.default_budgets.items()
            },
            "available": self.available,
            "unavailable_reason": self.unavailable_reason,
            "requirements": list(self.requirements),
        }


@dataclass(frozen=True, slots=True)
class WorkloadRequest:
    workload_id: str
    tier: WorkloadTier
    seed: int = 2026
    budget: WorkloadBudget | None = None
    warmup_runs: int = 1
    measured_runs: int = 3

    def __post_init__(self):
        object.__setattr__(self, "workload_id", nonempty(self.workload_id, "workload_id"))
        object.__setattr__(self, "tier", WorkloadTier(self.tier))
        seed = int(self.seed)
        if not 0 <= seed < 2**32:
            raise ValueError("seed must lie in [0, 2**32)")
        object.__setattr__(self, "seed", seed)
        if self.budget is not None and not isinstance(self.budget, WorkloadBudget):
            object.__setattr__(self, "budget", WorkloadBudget.from_dict(self.budget))
        warmups, measured = int(self.warmup_runs), int(self.measured_runs)
        if not 0 <= warmups <= 5:
            raise ValueError("warmup_runs must lie in [0,5]")
        if not 1 <= measured <= 9 or measured % 2 == 0:
            raise ValueError("measured_runs must be an odd integer in [1,9]")
        object.__setattr__(self, "warmup_runs", warmups)
        object.__setattr__(self, "measured_runs", measured)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "WorkloadRequest":
        data = dict(value)
        if data.get("budget") is not None:
            data["budget"] = WorkloadBudget.from_dict(data["budget"])
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        return {
            "workload_id": self.workload_id, "tier": self.tier.value,
            "seed": self.seed,
            "budget": None if self.budget is None else self.budget.to_dict(),
            "warmup_runs": self.warmup_runs,
            "measured_runs": self.measured_runs,
        }


@dataclass(frozen=True, slots=True)
class WorkloadQualification:
    qualification_id: str
    workload_id: str
    tier: WorkloadTier
    status: WorkloadStatus
    progress: float
    message: str
    request: dict[str, Any]
    budgets: dict[str, Any]
    metrics: dict[str, Any]
    comparison: dict[str, Any]
    evidence_relative_path: str
    content_hash: str
    process_id: int | None
    cancellation_requested: bool
    error: dict[str, Any] | None
    started_at: str
    finished_at: str
    created_at: str
    updated_at: str

    def __post_init__(self):
        for name in ("qualification_id", "workload_id", "message", "created_at", "updated_at"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        object.__setattr__(self, "tier", WorkloadTier(self.tier))
        object.__setattr__(self, "status", WorkloadStatus(self.status))
        progress = float(self.progress)
        if not 0 <= progress <= 1:
            raise ValueError("workload progress must lie in [0,1]")
        object.__setattr__(self, "progress", progress)
        for name in ("request", "budgets", "metrics", "comparison"):
            object.__setattr__(self, name, json_value(dict(getattr(self, name))))
        if self.error is not None:
            object.__setattr__(self, "error", json_value(dict(self.error)))
        digest = str(self.content_hash).lower()
        if digest and (len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("workload content_hash must be empty or SHA-256")
        object.__setattr__(self, "content_hash", digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualification_id": self.qualification_id,
            "workload_id": self.workload_id, "tier": self.tier.value,
            "status": self.status.value, "progress": self.progress,
            "message": self.message, "request": dict(self.request),
            "budgets": dict(self.budgets), "metrics": dict(self.metrics),
            "comparison": dict(self.comparison),
            "evidence_relative_path": self.evidence_relative_path,
            "content_hash": self.content_hash, "process_id": self.process_id,
            "cancellation_requested": self.cancellation_requested,
            "error": self.error, "started_at": self.started_at,
            "finished_at": self.finished_at, "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


__all__ = [
    "WorkloadBudget", "WorkloadDefinition", "WorkloadQualification",
    "WorkloadRequest", "WorkloadStatus", "WorkloadTier",
]
