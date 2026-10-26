"""Lazy result-state, dashboard, comparison, and export contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ._common import json_value, nonempty


@dataclass(frozen=True, slots=True)
class ResultField:
    name: str
    shape: tuple[int, ...]
    dtype: str
    minimum: float | None
    maximum: float | None
    mean: float | None
    nonfinite_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "shape": list(self.shape), "dtype": self.dtype,
            "minimum": self.minimum, "maximum": self.maximum, "mean": self.mean,
            "nonfinite_count": self.nonfinite_count,
        }


@dataclass(frozen=True, slots=True)
class ResultState:
    state_id: str
    run_id: str
    report_step: int
    simulated_time_seconds: float
    relative_path: str
    content_hash: str
    fields: tuple[ResultField, ...]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_id": self.state_id, "run_id": self.run_id,
            "report_step": self.report_step,
            "simulated_time_seconds": self.simulated_time_seconds,
            "relative_path": self.relative_path, "content_hash": self.content_hash,
            "fields": [item.to_dict() for item in self.fields], "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class RunComparison:
    comparison_id: str
    project_id: str
    reference_run_id: str
    candidate_run_id: str
    content_hash: str
    summary: Mapping[str, Any]
    created_at: str

    def __post_init__(self) -> None:
        for name in ("comparison_id", "project_id", "reference_run_id", "candidate_run_id", "created_at"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        object.__setattr__(self, "summary", json_value(dict(self.summary)))

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True)
class ResultExport:
    export_id: str
    run_id: str
    kind: str
    relative_path: str
    content_hash: str
    metadata: Mapping[str, Any]
    created_at: str

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}


__all__ = ["ResultExport", "ResultField", "ResultState", "RunComparison"]
