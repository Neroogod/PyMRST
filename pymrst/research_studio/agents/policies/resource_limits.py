"""Hard resource ceilings attached to an agent permission grant."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .._serialization import require_fields


@dataclass(frozen=True, slots=True)
class ExecutionBudget:
    max_cells: int = 1_000_000
    max_report_steps: int = 10_000
    max_realizations: int = 1
    max_workers: int = 1
    max_runtime_seconds: float = 3_600.0
    max_memory_bytes: int = 8 * 1024**3

    def __post_init__(self) -> None:
        integer_fields = (
            "max_cells", "max_report_steps", "max_realizations",
            "max_workers", "max_memory_bytes",
        )
        for name in integer_fields:
            value = int(getattr(self, name))
            if value <= 0:
                raise ValueError(f"{name} must be positive")
            object.__setattr__(self, name, value)
        runtime = float(self.max_runtime_seconds)
        if runtime <= 0.0:
            raise ValueError("max_runtime_seconds must be positive")
        object.__setattr__(self, "max_runtime_seconds", runtime)

    def to_dict(self) -> dict[str, Any]:
        return {
            "max_cells": self.max_cells,
            "max_report_steps": self.max_report_steps,
            "max_realizations": self.max_realizations,
            "max_workers": self.max_workers,
            "max_runtime_seconds": self.max_runtime_seconds,
            "max_memory_bytes": self.max_memory_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ExecutionBudget":
        return cls(**require_fields(
            value,
            required={
                "max_cells", "max_report_steps", "max_realizations",
                "max_workers", "max_runtime_seconds", "max_memory_bytes",
            },
            name="ExecutionBudget",
        ))


__all__ = ["ExecutionBudget"]
