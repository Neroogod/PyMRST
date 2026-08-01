"""Outline a coarse partition (MRST plotting/outlineCoarseGrid.m)."""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.plotting.plot_faces import plot_faces
from pymrst.plotting.results import PlotResult


DEFAULT_COARSE_COLOR = np.asarray([0.85, 0.15, 0.85])


def coarse_partition_faces(g, partition) -> np.ndarray:
    """Faces across partition changes, with the MRST boundary convention."""
    if g.griddim not in (2, 3):
        raise ValueError("outline_coarse_grid requires a 2-D or 3-D grid")
    raw = np.asarray(partition)
    if raw.ndim != 1 or raw.size != g.cells.num:
        raise ValueError(
            f"partition has {raw.size} entries; expected {g.cells.num}"
        )
    if not (np.issubdtype(raw.dtype, np.number) or raw.dtype == bool):
        raise TypeError("partition labels must be numeric")
    labels = np.asarray(raw, dtype=float)
    if not np.all(np.isfinite(labels)):
        raise ValueError("partition labels must be finite")

    neighbors = np.asarray(g.faces.neighbors, dtype=np.int64)
    mapped = np.full(neighbors.shape, np.nan, dtype=float)
    inside = neighbors >= 0
    mapped[inside] = labels[neighbors[inside]]
    different = (
        (inside[:, 0] != inside[:, 1])
        | (
            inside[:, 0] & inside[:, 1]
            & (mapped[:, 0] != mapped[:, 1])
        )
    )
    if g.griddim == 3:
        different &= np.all(inside, axis=1)
    return np.flatnonzero(different)


def outline_coarse_grid(
    g,
    partition,
    color: Any = DEFAULT_COARSE_COLOR,
    *,
    line_width: float = 2.0,
    plotter=None,
    show: bool = True,
    **mesh_kwargs,
) -> PlotResult:
    """Render faces/edges separating coarse partition blocks."""
    faces = coarse_partition_faces(g, partition)
    return plot_faces(
        g,
        faces,
        color,
        edge_color=color if g.griddim == 2 else "black",
        line_width=line_width,
        plotter=plotter,
        show=show,
        **mesh_kwargs,
    )


outlineCoarseGrid = outline_coarse_grid

__all__ = [
    "DEFAULT_COARSE_COLOR",
    "coarse_partition_faces",
    "outline_coarse_grid",
    "outlineCoarseGrid",
]
