"""Versioned report-boundary checkpoints and strict restart compatibility."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
from typing import Any, Callable

import numpy as np

from pymrst.research_studio.runtime.identifiers import CheckpointId, RunId

CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointCompatibilityError(RuntimeError):
    pass


def _canonical(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("fingerprint inputs must be finite")
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Enum):
        return _canonical(value.value)
    if isinstance(value, np.ndarray):
        return {
            "dtype": value.dtype.str,
            "shape": list(value.shape),
            "values": _canonical(value.tolist()),
        }
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [_canonical(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {
            "class": f"{type(value).__module__}.{type(value).__qualname__}",
            "fields": {field.name: _canonical(getattr(value, field.name)) for field in fields(value)},
        }
    raise TypeError(f"cannot fingerprint {type(value).__name__}; provide an explicit fingerprint")


def fingerprint(value: Any) -> str:
    payload = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def schedule_values(schedule: Any) -> np.ndarray:
    """Return a copied 1D timestep array for bare or structured schedules."""
    values = np.asarray(
        schedule.step_val if hasattr(schedule, "step_val") else schedule,
        dtype=float,
    ).reshape(-1)
    if np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError("schedule timesteps must be finite and positive")
    return values.copy()


def schedule_fingerprint(schedule: Any) -> str:
    if hasattr(schedule, "step_val"):
        return fingerprint({
            "kind": "structured",
            "step_val": np.asarray(schedule.step_val),
            "step_control": np.asarray(schedule.step_control),
            "controls": list(schedule.controls),
        })
    return fingerprint({"kind": "bare", "step_val": schedule_values(schedule)})


def slice_schedule(schedule: Any, start: int) -> Any:
    values = schedule_values(schedule)
    if start < 0 or start > values.size:
        raise CheckpointCompatibilityError("checkpoint is beyond the end of the schedule")
    if hasattr(schedule, "step_val"):
        from pymrst.modules.ad_core.utils.schedule import Schedule
        return Schedule(
            step_val=np.asarray(schedule.step_val)[start:].copy(),
            step_control=np.asarray(schedule.step_control)[start:].copy(),
            controls=list(schedule.controls),
        )
    return values[start:].copy()


def default_solver_fingerprint(solver: Any) -> str:
    names = (
        "max_iterations", "max_timestep_cuts", "use_line_search",
        "use_relaxation", "relaxation_min", "relaxation_increment",
        "line_search_max",
    )
    configuration = {
        "class": f"{type(solver).__module__}.{type(solver).__qualname__}",
        "options": {name: getattr(solver, name) for name in names if hasattr(solver, name)},
        "linear_solver": (type(getattr(solver, "linear_solver", None)).__qualname__
                          if getattr(solver, "linear_solver", None) is not None else None),
        "timestep_selector": (type(getattr(solver, "timestep_selector", None)).__qualname__
                              if getattr(solver, "timestep_selector", None) is not None else None),
    }
    return fingerprint(configuration)


def default_model_fingerprint(model: Any) -> str:
    return fingerprint({"class": f"{type(model).__module__}.{type(model).__qualname__}"})


@dataclass(frozen=True, slots=True)
class CheckpointRecord:
    checkpoint_id: CheckpointId
    source_run_id: RunId
    completed_report_steps: int
    accepted_elapsed_time: float
    state: Mapping[str, Any]
    schedule_fingerprint: str
    model_fingerprint: str
    solver_fingerprint: str
    code_fingerprint: str
    control_index: int | None = None
    schema_version: int = CHECKPOINT_SCHEMA_VERSION

    def __post_init__(self):
        if self.completed_report_steps < 0:
            raise ValueError("completed_report_steps must be non-negative")
        if self.accepted_elapsed_time < 0 or not np.isfinite(self.accepted_elapsed_time):
            raise ValueError("accepted_elapsed_time must be finite and non-negative")
        if not isinstance(self.state, Mapping):
            raise TypeError("checkpoint state must be an explicit mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "checkpoint_id": str(self.checkpoint_id),
            "source_run_id": str(self.source_run_id),
            "completed_report_steps": self.completed_report_steps,
            "accepted_elapsed_time": self.accepted_elapsed_time,
            "state": dict(self.state),
            "schedule_fingerprint": self.schedule_fingerprint,
            "model_fingerprint": self.model_fingerprint,
            "solver_fingerprint": self.solver_fingerprint,
            "code_fingerprint": self.code_fingerprint,
            "control_index": self.control_index,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CheckpointRecord":
        data = dict(value)
        if data.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointCompatibilityError(
                f"unsupported Checkpoint schema version {data.get('schema_version')!r}"
            )
        required = {
            "schema_version", "checkpoint_id", "source_run_id",
            "completed_report_steps", "accepted_elapsed_time", "state",
            "schedule_fingerprint", "model_fingerprint", "solver_fingerprint",
            "code_fingerprint",
            "control_index",
        }
        if set(data) != required:
            raise CheckpointCompatibilityError(
                f"invalid checkpoint fields: missing={sorted(required-set(data))}, "
                f"unknown={sorted(set(data)-required)}"
            )
        return cls(
            checkpoint_id=CheckpointId(data["checkpoint_id"]),
            source_run_id=RunId(data["source_run_id"]),
            completed_report_steps=int(data["completed_report_steps"]),
            accepted_elapsed_time=float(data["accepted_elapsed_time"]),
            state=data["state"],
            schedule_fingerprint=str(data["schedule_fingerprint"]),
            model_fingerprint=str(data["model_fingerprint"]),
            solver_fingerprint=str(data["solver_fingerprint"]),
            code_fingerprint=str(data["code_fingerprint"]),
            control_index=(int(data["control_index"])
                           if data["control_index"] is not None else None),
        )


def validate_checkpoint(
    checkpoint: CheckpointRecord,
    *,
    schedule: Any,
    model_fingerprint: str,
    solver_fingerprint: str,
    code_fingerprint: str,
) -> None:
    expected = {
        "schedule": (checkpoint.schedule_fingerprint, schedule_fingerprint(schedule)),
        "model": (checkpoint.model_fingerprint, model_fingerprint),
        "solver": (checkpoint.solver_fingerprint, solver_fingerprint),
        "code": (checkpoint.code_fingerprint, code_fingerprint),
    }
    mismatches = [name for name, (stored, current) in expected.items() if stored != current]
    if mismatches:
        raise CheckpointCompatibilityError(
            "checkpoint compatibility mismatch: " + ", ".join(mismatches)
        )
    if checkpoint.completed_report_steps > schedule_values(schedule).size:
        raise CheckpointCompatibilityError("checkpoint is beyond the end of the schedule")


def restart_payload(
    checkpoint: CheckpointRecord,
    *,
    schedule: Any,
    model_fingerprint: str,
    solver_fingerprint: str,
    code_fingerprint: str,
    state_factory: Callable[[Mapping[str, Any]], Any] = dict,
) -> tuple[Any, np.ndarray]:
    validate_checkpoint(
        checkpoint,
        schedule=schedule,
        model_fingerprint=model_fingerprint,
        solver_fingerprint=solver_fingerprint,
        code_fingerprint=code_fingerprint,
    )
    return (
        state_factory(checkpoint.state),
        slice_schedule(schedule, checkpoint.completed_report_steps),
    )


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION", "CheckpointCompatibilityError",
    "CheckpointRecord", "default_model_fingerprint",
    "default_solver_fingerprint", "fingerprint", "restart_payload",
    "schedule_fingerprint", "schedule_values", "slice_schedule",
    "validate_checkpoint",
]
