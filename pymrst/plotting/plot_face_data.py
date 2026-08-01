"""Plot boundary face data (MRST ``plotting/plotFaceData.m``)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from pymrst.plotting.boundary_faces import boundary_faces
from pymrst.plotting.plot_faces import plot_faces
from pymrst.plotting.private.coarse_faces import coarse_grid_parts
from pymrst.plotting.results import PlotResult


def _empty_result(g, plotter=None) -> PlotResult:
    from pymrst.plotting.private.plot_patches import build_patch_geometry

    parts = coarse_grid_parts(g)
    entity_grid = parts[0] if parts is not None else g
    return PlotResult(
        geometry=build_patch_geometry(entity_grid, []),
        plotter=plotter,
    )


def _face_data(g, data: Any) -> np.ndarray:
    values = np.asarray(data)
    if values.ndim == 0:
        raise ValueError("face data must contain one row per grid face")
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] not in (1, 3):
        raise ValueError("face data must have one scalar or three RGB columns")
    if values.shape[0] != g.faces.num:
        raise ValueError(
            f"face data has {values.shape[0]} rows; expected {g.faces.num}"
        )
    return values


def plot_face_data(
    g,
    *args,
    plotter=None,
    filename: str | Path | None = None,
    show: bool = True,
    **mesh_kwargs,
) -> PlotResult:
    """Plot face-centred values on the boundary of a cell subset.

    Supported call forms mirror MRST, with zero-based Python indices::

        plot_face_data(grid, face_data)
        plot_face_data(grid, cells, face_data)

    ``face_data`` must contain one scalar or RGB row for every face in
    ``grid``. A logical ``cells`` mask is also accepted.
    """
    if len(args) == 1:
        cells = None
        data = args[0]
    elif len(args) == 2:
        cells, data = args
    else:
        raise TypeError(
            "plot_face_data expects (grid, data) or (grid, cells, data)"
        )

    parts = coarse_grid_parts(g)
    entity_grid = parts[0] if parts is not None else g
    selected_cells = (
        np.arange(entity_grid.cells.num, dtype=np.int64)
        if cells is None else np.asarray(cells)
    )
    if selected_cells.size == 0:
        return _empty_result(g, plotter)
    if entity_grid.griddim != 3:
        raise ValueError("plot_face_data requires a 3-D grid")

    values = _face_data(entity_grid, data)
    faces = boundary_faces(entity_grid, selected_cells)
    return plot_faces(
        g,
        faces,
        values[faces],
        edge_color=None,
        plotter=plotter,
        filename=filename,
        show=show,
        **mesh_kwargs,
    )


plotFaceData = plot_face_data

__all__ = ["plot_face_data", "plotFaceData"]
