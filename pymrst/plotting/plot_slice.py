"""Plot Cartesian cell-data slices (MRST plotting/plotSlice.m)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pymrst.plotting.plot_faces import plot_faces
from pymrst.plotting.results import PlotResult


@dataclass
class SlicePlotResult:
    dimension: int
    slice_index: int
    cells: np.ndarray
    faces: np.ndarray
    values: np.ndarray
    view_vector: np.ndarray
    rendered: PlotResult

    @property
    def actor(self):
        return self.rendered.actor

    @property
    def geometry(self):
        return self.rendered.geometry


def _slice_cartesian_cells(g, slice_index: int, dimension: int) -> np.ndarray:
    if g.cartDims is None or len(g.cartDims) != 3:
        raise ValueError("plot_slice requires three-dimensional cartDims")
    if dimension not in (1, 2, 3):
        raise ValueError("dimension must be 1, 2, or 3")
    axis = dimension-1
    dims = tuple(map(int, g.cartDims))
    if slice_index < 0 or slice_index >= dims[axis]:
        raise IndexError(
            f"slice_index must lie in [0, {dims[axis]-1}] for dimension "
            f"{dimension}"
        )
    coordinates = np.indices(dims)
    selector = [slice(None)]*3
    selector[axis] = slice_index
    ijk = [coordinates[d][tuple(selector)].ravel(order="F") for d in range(3)]
    cartesian = np.ravel_multi_index(ijk, dims, order="F")
    inverse = np.full(int(np.prod(dims)), -1, dtype=np.int64)
    index_map = np.asarray(g.cells.indexMap, dtype=np.int64).ravel()
    inverse[index_map] = np.arange(g.cells.num, dtype=np.int64)
    active = inverse[cartesian]
    return active[active >= 0]


def slice_entities(g, slice_index: int, dimension: int):
    """Return active cells and negative-side tagged faces for a slice."""
    cells = _slice_cartesian_cells(g, int(slice_index), int(dimension))
    target_tag = (0, 2, 4)[dimension-1]
    faces = []
    owners = []
    tags = getattr(g.cells, "tags", None)
    if tags is None:
        raise ValueError("plot_slice requires directional cell-face tags")
    tags = np.asarray(tags, dtype=np.int64).ravel()
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64)
    cell_faces = cell_faces[:, 0] if cell_faces.ndim > 1 else cell_faces
    for cell in cells:
        lo, hi = g.cells.facePos[cell:cell+2]
        matches = np.flatnonzero(tags[lo:hi] == target_tag)+lo
        faces.extend(cell_faces[matches].tolist())
        owners.extend([int(cell)]*matches.size)
    return (
        cells,
        np.asarray(faces, dtype=np.int64),
        np.asarray(owners, dtype=np.int64),
    )


def plot_slice(
    g,
    data: Any,
    slice_index: int,
    dimension: int,
    *,
    plotter=None,
    show: bool = True,
    **mesh_kwargs,
) -> SlicePlotResult:
    """Plot one zero-based Cartesian slice in dimension 1, 2, or 3."""
    values = np.asarray(data)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] not in (1, 3):
        raise ValueError("slice data must have one scalar or three RGB columns")
    if values.shape[0] != g.cells.num:
        raise ValueError(
            f"slice data has {values.shape[0]} rows; expected {g.cells.num}"
        )
    cells, faces, owners = slice_entities(g, slice_index, dimension)
    face_values = values[owners]
    rendered = plot_faces(
        g,
        faces,
        face_values,
        plotter=plotter,
        show=show,
        **mesh_kwargs,
    )
    view = np.zeros(3, dtype=float)
    view[dimension-1] = 1.0
    return SlicePlotResult(
        dimension=int(dimension),
        slice_index=int(slice_index),
        cells=cells,
        faces=faces,
        values=face_values,
        view_vector=view,
        rendered=rendered,
    )


plotSlice = plot_slice

__all__ = ["SlicePlotResult", "slice_entities", "plot_slice", "plotSlice"]
