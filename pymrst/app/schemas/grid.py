"""Serializable grid and property inspection contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ._common import json_value, nonempty


@dataclass(frozen=True, slots=True)
class GridArraySummary:
    name: str
    shape: tuple[int, ...]
    dtype: str
    minimum: float | None
    maximum: float | None
    mean: float | None
    percentiles: Mapping[str, float]
    nonfinite_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", nonempty(self.name, "array name"))
        shape = tuple(int(item) for item in self.shape)
        if any(item < 0 for item in shape):
            raise ValueError("array shape cannot be negative")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "dtype", nonempty(self.dtype, "array dtype"))
        object.__setattr__(self, "percentiles", json_value(dict(self.percentiles)))
        object.__setattr__(self, "nonfinite_count", int(self.nonfinite_count))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "shape": list(self.shape), "dtype": self.dtype,
            "minimum": self.minimum, "maximum": self.maximum, "mean": self.mean,
            "percentiles": dict(self.percentiles),
            "nonfinite_count": self.nonfinite_count,
        }


@dataclass(frozen=True, slots=True)
class GridSnapshot:
    snapshot_id: str
    project_id: str
    model_id: str
    name: str
    content_hash: str
    cell_count: int
    face_count: int
    node_count: int
    dimension: int
    active_cell_count: int
    bounds: tuple[tuple[float, float], ...]
    arrays: tuple[GridArraySummary, ...]
    findings: tuple[Mapping[str, Any], ...]
    created_at: str
    # How to deterministically rebuild the full grid topology (faces, nodes,
    # connectivity) that this snapshot's summary arrays were derived from.
    # The snapshot itself only stores bounded arrays for UI inspection, not
    # enough to reconstruct a runnable grid — {"kind": "unknown"} for grids
    # registered without a recorded recipe (e.g. directly in tests).
    provenance: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id, "project_id": self.project_id,
            "model_id": self.model_id, "name": self.name,
            "content_hash": self.content_hash, "cell_count": self.cell_count,
            "face_count": self.face_count, "node_count": self.node_count,
            "dimension": self.dimension, "active_cell_count": self.active_cell_count,
            "bounds": [list(item) for item in self.bounds],
            "arrays": [item.to_dict() for item in self.arrays],
            "findings": [dict(item) for item in self.findings],
            "created_at": self.created_at,
            "provenance": dict(self.provenance),
        }


__all__ = ["GridArraySummary", "GridSnapshot"]
