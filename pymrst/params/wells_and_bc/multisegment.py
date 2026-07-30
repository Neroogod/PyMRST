"""MRST-style multisegment well data containers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class WellSegment:
    index: int
    parent: int
    length: float = 0.0
    depth: float = 0.0
    diameter: float = 0.0
    roughness: float = 0.0
    volume: float = 0.0
    flow_model: str = "tubing"
    valve_diameter: float = 0.0
    discharge_coefficient: float = 1.0
    valve_count: int = 1


@dataclass
class MultisegmentWell:
    """Topology and geometry for MRST ``MultisegmentWell`` schedules."""

    segments: list[WellSegment] = field(default_factory=list)
    perforation_segment: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @classmethod
    def from_arrays(cls, parents, lengths=None, depths=None, diameters=None,
                    roughness=None, volumes=None, perforation_segment=None):
        parents = np.asarray(parents, dtype=np.int64).ravel()
        n = parents.size
        lengths = np.zeros(n) if lengths is None else np.asarray(lengths, dtype=float).ravel()
        depths = np.zeros(n) if depths is None else np.asarray(depths, dtype=float).ravel()
        diameters = np.zeros(n) if diameters is None else np.asarray(diameters, dtype=float).ravel()
        roughness = np.zeros(n) if roughness is None else np.asarray(roughness, dtype=float).ravel()
        volumes = np.zeros(n) if volumes is None else np.asarray(volumes, dtype=float).ravel()
        segs = [
            WellSegment(i, int(parents[i]), float(lengths[i]), float(depths[i]),
                        float(diameters[i]), float(roughness[i]), float(volumes[i]))
            for i in range(n)
        ]
        perf = (
            np.zeros(0, dtype=np.int64)
            if perforation_segment is None
            else np.asarray(perforation_segment, dtype=np.int64).ravel()
        )
        return cls(segs, perf)

    def path_to_root(self, segment: int) -> np.ndarray:
        path = []
        seen = set()
        current = int(segment)
        while current >= 0:
            if current in seen:
                raise ValueError("cycle in multisegment well topology")
            seen.add(current)
            path.append(current)
            parent = self.segments[current].parent
            current = parent
        return np.asarray(path, dtype=np.int64)

    def validate(self, perforation_count: int | None = None) -> None:
        """Validate the rooted segment tree and completion-node mapping."""
        if not self.segments:
            raise ValueError("a multisegment well needs at least one node")
        indices = [segment.index for segment in self.segments]
        if indices != list(range(len(self.segments))):
            raise ValueError("segment indices must be contiguous and zero-based")
        if self.segments[0].parent != -1:
            raise ValueError("the root segment must have parent -1")
        for segment in self.segments[1:]:
            if not 0 <= segment.parent < segment.index:
                raise ValueError("each segment parent must precede its child")
            if segment.flow_model not in {"tubing", "valve", "none"}:
                raise ValueError(f"unknown segment flow model {segment.flow_model!r}")
            if segment.flow_model == "valve":
                if segment.valve_diameter <= 0 or segment.discharge_coefficient <= 0:
                    raise ValueError("valve diameter and discharge coefficient must be positive")
                if segment.valve_count < 1:
                    raise ValueError("valve_count must be positive")
        if perforation_count is not None and self.perforation_segment.size != perforation_count:
            raise ValueError("one perforation_segment entry is required per completion")
        if self.perforation_segment.size and np.any(
            (self.perforation_segment < 0) |
            (self.perforation_segment >= len(self.segments))
        ):
            raise ValueError("perforation segment lies outside the node graph")


__all__ = ["MultisegmentWell", "WellSegment"]
