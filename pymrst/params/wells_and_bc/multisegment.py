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


__all__ = ["MultisegmentWell", "WellSegment"]
