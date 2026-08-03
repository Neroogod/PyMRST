"""Storage protocol independent of HDF5, Zarr, or remote implementations."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from pymrst.research_studio.evidence.report import SimulationEvidence
from pymrst.research_studio.evidence.events import SimulationEvent
from pymrst.research_studio.runtime.identifiers import CheckpointId, RunId


class StoreCorruptionError(RuntimeError):
    pass


@runtime_checkable
class ResultStore(Protocol):
    def begin_run(self, run_id: RunId, manifest: Mapping[str, Any]) -> None: ...
    def commit_state(self, run_id: RunId, index: int, state: Mapping[str, Any], *, accepted: bool = True) -> None: ...
    def commit_evidence(self, run_id: RunId, evidence: SimulationEvidence) -> None: ...
    def commit_event(self, run_id: RunId, event: SimulationEvent) -> None: ...
    def commit_checkpoint(self, run_id: RunId, checkpoint_id: CheckpointId, data: Mapping[str, Any]) -> None: ...
    def load_state(self, run_id: RunId, index: int) -> dict[str, Any]: ...
    def load_checkpoint(self, run_id: RunId, checkpoint_id: CheckpointId) -> dict[str, Any]: ...
    def finalize_run(self, run_id: RunId, status: str) -> None: ...


__all__ = ["ResultStore", "StoreCorruptionError"]
