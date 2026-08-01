"""Plot grid subsets (MRST ``plotting/plotGrid.m``)."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import numpy as np

from pymrst.plotting.boundary_faces import boundary_faces
from pymrst.plotting.plot_cell_data import (
    _cell_indices,
    _plot_1d_series,
    plot_cell_data,
)
from pymrst.plotting.plot_faces import plot_faces
from pymrst.plotting.private.coarse_faces import coarse_grid_parts
from pymrst.plotting.plot_well import plot_well, plotWell


def _rgb(color) -> np.ndarray:
    if not isinstance(color, str):
        values = np.asarray(color, dtype=float).ravel()
        if values.size != 3:
            raise ValueError("grid color must be a name or RGB triplet")
        return values
    try:
        from matplotlib.colors import to_rgb
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "named plot colors require Matplotlib; "
            "install pymrst[plotting]"
        ) from error
    return np.asarray(to_rgb(color), dtype=float)


def plot_grid(
    g,
    cells=None,
    color="yellow",
    *,
    face_color=None,
    edge_color="black",
    show_edges: bool | None = None,
    opacity: float = 1.0,
    line_width: float = 1.0,
    plotter=None,
    filename: str | Path | None = None,
    show: bool = True,
    flip_z: bool = True,
    zscale: float = 1.0,
    **mesh_kwargs,
):
    """Plot the exterior of a numeric/logical grid-cell subset."""
    parts = coarse_grid_parts(g)
    entity_grid = parts[0] if parts is not None else g
    selected = _cell_indices(entity_grid, cells)
    if show_edges is not None:
        edge_color = "black" if show_edges else None
    effective = color if face_color is None else face_color

    if entity_grid.griddim == 3:
        faces = boundary_faces(entity_grid, selected)
        return plot_faces(
            g,
            faces,
            effective,
            face_color=effective,
            edge_color=edge_color,
            opacity=opacity,
            line_width=line_width,
            flip_z=flip_z,
            zscale=zscale,
            plotter=plotter,
            filename=filename,
            show=show,
            **mesh_kwargs,
        )
    if entity_grid.griddim == 2:
        rgb = np.tile(_rgb(color), (selected.size, 1))
        if isinstance(effective, str) and effective.lower() == "none":
            mesh_kwargs.setdefault("style", "wireframe")
        elif effective is not None:
            rgb = np.tile(_rgb(effective), (selected.size, 1))
        return plot_cell_data(
            g,
            rgb,
            selected,
            edge_color=edge_color,
            opacity=opacity,
            flip_z=flip_z,
            zscale=zscale,
            line_width=line_width,
            plotter=plotter,
            filename=filename,
            show=show,
            **mesh_kwargs,
        )
    if entity_grid.griddim == 1:
        if parts is not None:
            raise NotImplementedError(
                "coarse 1-D plotGrid is not represented by PyMRST grids"
            )
        mesh_kwargs.setdefault("color", color)
        mesh_kwargs.setdefault("line_width", float(line_width))
        return _plot_1d_series(
            entity_grid,
            selected,
            np.ones((selected.size, 1), dtype=float),
            plotter=plotter,
            filename=filename,
            show=show,
            opacity=opacity,
            mesh_kwargs=mesh_kwargs,
        )
    raise ValueError(f"unsupported grid dimension {entity_grid.griddim}")


plotGrid = plot_grid

_LEGACY = {"to_pyvista", "_finish"}


def __getattr__(name):
    """Load historical PyVista helpers only when explicitly requested."""
    if name not in _LEGACY:
        raise AttributeError(name)
    value = getattr(import_module("pymrst.plotting._pyvista_grid"), name)
    globals()[name] = value
    return value


__all__ = [
    "plot_grid",
    "plotGrid",
    "plot_cell_data",
    "to_pyvista",
    "plot_well",
    "plotWell",
]
