"""Plot nodal data (MRST ``plotting/plotNodeData.m``)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from pymrst.plotting.boundary_faces import boundary_faces
from pymrst.plotting.plot_cell_data import _cell_indices, _cell_patch_geometry
from pymrst.plotting.plot_faces import plot_faces
from pymrst.plotting.private.coarse_faces import coarse_grid_parts
from pymrst.plotting.results import PatchGeometry, PlotResult
from pymrst.plotting.private.rendering_compat import (
    translate_matlab_properties,
)


def _node_values(g, data: Any) -> np.ndarray:
    values = np.asarray(data)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] not in (1, 3):
        raise ValueError("node data must have one scalar or three RGB columns")
    if values.shape[0] != g.nodes.num:
        raise ValueError(
            f"node data has {values.shape[0]} rows; expected {g.nodes.num}"
        )
    return values


def _displaced_grid(g, uu):
    coords = np.asarray(g.nodes.coords, dtype=float)
    if uu is None:
        return g
    displacement = np.asarray(uu, dtype=float)
    if displacement.shape != coords.shape:
        raise ValueError(
            f"uu has shape {displacement.shape}; expected {coords.shape}"
        )
    return replace(
        g,
        nodes=replace(g.nodes, coords=coords + displacement),
    )


def _render_2d_nodes(
    g,
    geometry: PatchGeometry,
    values: np.ndarray,
    *,
    plotter,
    filename,
    show,
    cmap,
    clim,
    edge_color,
    opacity,
    mesh_kwargs,
) -> PlotResult:
    try:
        import pyvista as pv
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "plot_node_data requires PyVista; install pymrst[plotting]"
        ) from error
    own_plotter = plotter is None
    target = plotter or pv.Plotter(off_screen=filename is not None)
    result = PlotResult(geometry=geometry, plotter=target)
    if geometry.n_entities:
        points = geometry.vertices
        if points.shape[1] == 2:
            z = getattr(g.nodes, "z", None)
            lifted = (
                np.asarray(z, dtype=float).ravel()[geometry.node_ids]
                if z is not None else np.zeros(points.shape[0])
            )
            points = np.column_stack([points, lifted])
            geometry = replace(geometry, vertices=points)
            result.geometry = geometry
        stream = np.concatenate([
            np.r_[loop.size, loop] for loop in geometry.connectivity
        ]).astype(np.int64)
        mesh = pv.PolyData(points, faces=stream)
        mesh.field_data["grid_cell_ids"] = geometry.entity_ids
        nodal = values[geometry.node_ids]
        mesh.point_data["node_data"] = (
            nodal[:, 0] if nodal.shape[1] == 1 else nodal
        )
        kwargs = translate_matlab_properties(mesh_kwargs)
        kwargs.setdefault("scalars", "node_data")
        kwargs.setdefault("preference", "point")
        kwargs.setdefault("opacity", float(opacity))
        kwargs.setdefault("show_edges", edge_color is not None)
        if edge_color is not None:
            kwargs.setdefault("edge_color", edge_color)
        if nodal.shape[1] == 3:
            kwargs.setdefault("rgb", True)
        else:
            kwargs.setdefault("cmap", cmap)
            if clim is not None:
                kwargs.setdefault("clim", clim)
        result.actor = target.add_mesh(mesh, **kwargs)
    if filename is not None:
        target.screenshot(str(filename))
        if own_plotter:
            target.close()
    elif show and own_plotter:
        target.show()
    return result


def plot_node_data(
    g,
    node_data: Any,
    cells=None,
    *,
    uu=None,
    plotter=None,
    filename: str | Path | None = None,
    show: bool = True,
    cmap: str = "viridis",
    clim=None,
    edge_color="black",
    opacity: float = 1.0,
    flip_z: bool = True,
    zscale: float = 1.0,
    **mesh_kwargs,
) -> PlotResult:
    """Plot scalar or RGB values interpolated across grid vertices."""
    if coarse_grid_parts(g) is not None:
        raise NotImplementedError(
            "plot_node_data requires a fine grid with explicit nodes"
        )
    selected = _cell_indices(g, cells)
    values = _node_values(g, node_data)
    render_grid = _displaced_grid(g, uu)

    if g.griddim == 3:
        faces = boundary_faces(g, selected)
        return plot_faces(
            render_grid,
            faces,
            values,
            edge_color=edge_color,
            opacity=opacity,
            flip_z=flip_z,
            zscale=zscale,
            cmap=cmap,
            clim=clim,
            plotter=plotter,
            filename=filename,
            show=show,
            **mesh_kwargs,
        )
    if g.griddim == 2:
        geometry = _cell_patch_geometry(render_grid, selected)
        return _render_2d_nodes(
            render_grid,
            geometry,
            values,
            plotter=plotter,
            filename=filename,
            show=show,
            cmap=cmap,
            clim=clim,
            edge_color=edge_color,
            opacity=opacity,
            mesh_kwargs=mesh_kwargs,
        )
    raise NotImplementedError("1-D plotNodeData rendering is not yet ported")


plotNodeData = plot_node_data

__all__ = ["plot_node_data", "plotNodeData"]
