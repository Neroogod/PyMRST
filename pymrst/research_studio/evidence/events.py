"""Immutable lifecycle events shared by renderers and evidence stores."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Mapping

from pymrst.research_studio.runtime.identifiers import (
    AttemptId, MinistepId, ReportStepId, RunId,
)


class EventKind(StrEnum):
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"
    REPORT_STEP_STARTED = "report_step_started"
    MINISTEP_STARTED = "ministep_started"
    NONLINEAR_ITERATION_COMPLETED = "nonlinear_iteration_completed"
    LINEAR_SOLVE_COMPLETED = "linear_solve_completed"
    CONTROL_SWITCHED = "control_switched"
    STEP_ACCEPTED = "step_accepted"
    STEP_REJECTED = "step_rejected"
    TIMESTEP_CUT = "timestep_cut"
    INVARIANT_VIOLATED = "invariant_violated"
    CHECKPOINT_COMMITTED = "checkpoint_committed"
    FAILURE_BUNDLE_WRITTEN = "failure_bundle_written"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, frozenset)):
        return [_plain(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class SimulationEvent:
    """One immutable observation of a simulation lifecycle transition."""

    kind: EventKind
    run_id: RunId
    sequence: int
    elapsed_seconds: float
    report_step_id: ReportStepId | None = None
    ministep_id: MinistepId | None = None
    attempt_id: AttemptId | None = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("event sequence must be non-negative")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")
        object.__setattr__(self, "payload", _freeze(self.payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "run_id": str(self.run_id),
            "sequence": self.sequence,
            "elapsed_seconds": self.elapsed_seconds,
            "report_step_id": (str(self.report_step_id)
                               if self.report_step_id is not None else None),
            "ministep_id": (str(self.ministep_id)
                            if self.ministep_id is not None else None),
            "attempt_id": (str(self.attempt_id)
                           if self.attempt_id is not None else None),
            "payload": _plain(self.payload),
        }


__all__ = ["EventKind", "SimulationEvent"]
