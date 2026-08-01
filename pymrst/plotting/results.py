"""Backend-neutral geometry and result objects for MRST-shaped plotting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PatchGeometry:
    """Compact topology for a selected set of grid faces.

    ``connectivity`` contains local vertex indices into ``vertices``. Each
    tuple entry is one selected face (or one edge for a 2-D grid).
    """

    vertices: np.ndarray
    connectivity: tuple[np.ndarray, ...]
    entity_ids: np.ndarray
    node_ids: np.ndarray
    dimension: int
    group_ids: np.ndarray | None = None

    @property
    def n_entities(self) -> int:
        return int(self.entity_ids.size)


@dataclass(frozen=True)
class ColorData:
    """Normalized color values and where they live."""

    mode: str
    values: Any


@dataclass
class PlotResult:
    """Rendered backend handle together with the geometry that produced it."""

    geometry: PatchGeometry
    actor: Any = None
    plotter: Any = None
    outline_actor: Any = None


__all__ = ["ColorData", "PatchGeometry", "PlotResult"]
