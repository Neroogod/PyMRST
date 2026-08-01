"""Plot scalar data over a 2-D unstructured grid as a lifted surface."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from pymrst.plotting.interpolate_data import interpolate_data
from pymrst.plotting.plot_cell_data import _cell_indices, _cell_patch_geometry
from pymrst.plotting.results import PatchGeometry, PlotResult
from pymrst.plotting.private.rendering_compat import (
    translate_matlab_properties,
)


@dataclass
class SurfacePlotResult:
    geometry: PatchGeometry
    node_values: np.ndarray
    rendered: PlotResult

    @property
    def actor(self):
        return self.rendered.actor

    @property
    def plotter(self):
        return self.rendered.plotter


def surface_geometry(g, data: Any, cells=None, *, extrapolation="linear"):
    """Return selected lifted polygons and full-grid nodal values."""
    if g.griddim != 2:
        raise ValueError("unstructured_surf supports only 2-D grids")
    selected = _cell_indices(g, cells)
    values = np.asarray(data, dtype=float)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError("unstructured_surf requires scalar data")
    if values.size == g.nodes.num:
        nodal = values.copy()
    else:
        nodal = interpolate_data(
            g,
            values,
            xq="nodes",
            interpolation="linear",
            extrapolation=extrapolation,
        )
    geometry = _cell_patch_geometry(g, selected)
    lifted = np.column_stack([
        geometry.vertices[:, :2], nodal[geometry.node_ids]
    ])
    return replace(geometry, vertices=lifted), nodal


def unstructured_surf(
    g,
    data: Any,
    cells=None,
    *,
    extrapolation: str = "linear",
    plotter=None,
    filename: str | Path | None = None,
    show: bool = True,
    cmap: str = "viridis",
    clim=None,
    edge_color="black",
    opacity: float = 1.0,
    **mesh_kwargs,
) -> SurfacePlotResult:
    """Render data as z-coordinates and interpolated nodal colors."""
    geometry, nodal = surface_geometry(
        g, data, cells, extrapolation=extrapolation
    )
    try:
        import pyvista as pv
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "unstructured_surf requires PyVista; install pymrst[plotting]"
        ) from error
    own_plotter = plotter is None
    target = plotter or pv.Plotter(off_screen=filename is not None)
    rendered = PlotResult(geometry=geometry, plotter=target)
    if geometry.n_entities:
        stream = np.concatenate([
            np.r_[loop.size, loop] for loop in geometry.connectivity
        ]).astype(np.int64)
        mesh = pv.PolyData(geometry.vertices, faces=stream)
        mesh.field_data["grid_cell_ids"] = geometry.entity_ids
        mesh.point_data["surface_data"] = nodal[geometry.node_ids]
        kwargs = translate_matlab_properties(mesh_kwargs)
        kwargs.setdefault("scalars", "surface_data")
        kwargs.setdefault("preference", "point")
        kwargs.setdefault("cmap", cmap)
        kwargs.setdefault("opacity", float(opacity))
        kwargs.setdefault("show_edges", edge_color is not None)
        if edge_color is not None:
            kwargs.setdefault("edge_color", edge_color)
        if clim is not None:
            kwargs.setdefault("clim", clim)
        rendered.actor = target.add_mesh(mesh, **kwargs)
    if filename is not None:
        target.screenshot(str(filename))
        if own_plotter:
            target.close()
    elif show and own_plotter:
        target.show()
    return SurfacePlotResult(geometry, nodal, rendered)


unstructuredSurf = unstructured_surf

__all__ = [
    "SurfacePlotResult",
    "surface_geometry",
    "unstructured_surf",
    "unstructuredSurf",
]
