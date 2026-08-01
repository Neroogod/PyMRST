"""Plot a coarse block and neighbors (MRST plotBlockAndNeighbors.m)."""

from __future__ import annotations

from colorsys import hsv_to_rgb
from dataclasses import dataclass
from typing import Any

import numpy as np

from pymrst.plotting.boundary_faces import boundary_faces
from pymrst.plotting.plot_faces import plot_faces
from pymrst.plotting.private.coarse_faces import coarse_grid_parts
from pymrst.plotting.results import PlotResult


@dataclass
class BlockNeighborsResult:
    block: int
    neighbors: np.ndarray
    block_faces: np.ndarray
    neighbor_faces: tuple[np.ndarray, ...]
    block_fault_faces: np.ndarray
    neighbor_fault_faces: tuple[np.ndarray, ...]
    block_actor: PlotResult
    neighbor_actors: tuple[PlotResult, ...]
    fault_actors: tuple[PlotResult, ...]
    plotter: Any = None


def coarse_block_neighbors(coarse, block: int) -> np.ndarray:
    if block < 0 or block >= coarse.cells.num:
        raise IndexError(f"block must lie in [0, {coarse.cells.num-1}]")
    neighbors = np.asarray(coarse.faces.neighbors, dtype=np.int64)
    incident = neighbors[np.any(neighbors == block, axis=1)].ravel()
    return np.unique(incident[(incident >= 0) & (incident != block)])


def _fault_faces(parent, partition, block) -> np.ndarray:
    tag = getattr(parent.faces, "tag", None)
    if tag is None:
        return np.empty(0, dtype=np.int64)
    neighbors = np.asarray(parent.faces.neighbors, dtype=np.int64)
    touches = np.zeros(parent.faces.num, dtype=bool)
    for side in range(2):
        inside = neighbors[:, side] >= 0
        touches[inside] |= partition[neighbors[inside, side]] == block
    return np.flatnonzero(touches & (np.asarray(tag).ravel() > 0))


def _neighbor_colors(count: int) -> tuple[tuple[float, float, float], ...]:
    total = max(8, count)
    base = [hsv_to_rgb(index/total, 1.0, 1.0) for index in range(total)]
    return tuple(tuple((2.0*np.asarray(rgb)+1.0)/3.0) for rgb in base[:count])


def plot_block_and_neighbors(
    coarse_grid,
    block: int,
    *,
    plot_faults=(True, True),
    alpha=(1.0, 1.0, 1.0, 1.0),
    block_color="blue",
    plotter=None,
    show: bool = True,
    **mesh_kwargs,
) -> BlockNeighborsResult:
    """Plot one zero-based coarse block and all face-neighbor blocks."""
    parts = coarse_grid_parts(coarse_grid)
    if parts is None:
        raise ValueError("plot_block_and_neighbors requires a coarse grid")
    coarse, parent, partition = parts
    if parent.griddim != 3:
        raise ValueError("plot_block_and_neighbors requires a 3-D grid")
    block = int(block)
    neighbors = coarse_block_neighbors(coarse, block)
    fault_modes = np.asarray(plot_faults, dtype=bool).ravel()
    if fault_modes.size != 2:
        raise ValueError("plot_faults must contain block and neighbor flags")
    opacity = np.asarray(alpha, dtype=float).ravel()
    required = 2 + (2 if fault_modes[1] else 1 if fault_modes[0] else 0)
    if opacity.size < required:
        raise ValueError(f"alpha must contain at least {required} values")
    if np.any((opacity < 0) | (opacity > 1)):
        raise ValueError("alpha values must lie in [0, 1]")

    target = plotter
    fault_results = []

    def render(one_block, color, surface_alpha, include_faults, fault_alpha):
        nonlocal target
        surface = boundary_faces(parent, partition == one_block)
        faults = _fault_faces(parent, partition, one_block) if include_faults else (
            np.empty(0, dtype=np.int64)
        )
        visible = np.setdiff1d(surface, faults, assume_unique=False)
        if faults.size:
            fault_result = plot_faces(
                parent,
                faults,
                [0.7, 0.7, 0.7],
                edge_color="red",
                opacity=float(fault_alpha),
                plotter=target,
                show=False,
                **mesh_kwargs,
            )
            target = fault_result.plotter
            fault_results.append(fault_result)
        result = plot_faces(
            parent,
            visible,
            color,
            opacity=float(surface_alpha),
            plotter=target,
            show=False,
            **mesh_kwargs,
        )
        target = result.plotter
        return visible, faults, result

    block_faces, block_fault_faces, block_result = render(
        block,
        block_color,
        opacity[0],
        fault_modes[0],
        opacity[2] if fault_modes[0] else 1.0,
    )
    neighbor_faces = []
    neighbor_fault_faces = []
    neighbor_results = []
    colors = _neighbor_colors(neighbors.size)
    for ordinal, neighbor in enumerate(neighbors):
        visible, faults, result = render(
            int(neighbor),
            colors[ordinal],
            opacity[1],
            fault_modes[1],
            opacity[3] if fault_modes[1] else 1.0,
        )
        neighbor_faces.append(visible)
        neighbor_fault_faces.append(faults)
        neighbor_results.append(result)
    if show and plotter is None and target is not None:
        target.show()
    return BlockNeighborsResult(
        block=block,
        neighbors=neighbors,
        block_faces=block_faces,
        neighbor_faces=tuple(neighbor_faces),
        block_fault_faces=block_fault_faces,
        neighbor_fault_faces=tuple(neighbor_fault_faces),
        block_actor=block_result,
        neighbor_actors=tuple(neighbor_results),
        fault_actors=tuple(fault_results),
        plotter=target,
    )


plotBlockAndNeighbors = plot_block_and_neighbors

__all__ = [
    "BlockNeighborsResult",
    "coarse_block_neighbors",
    "plot_block_and_neighbors",
    "plotBlockAndNeighbors",
]
