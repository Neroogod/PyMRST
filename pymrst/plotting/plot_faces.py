"""Plot selected grid faces (MRST ``plotting/plotFaces.m``)."""

from __future__ import annotations

from pathlib import Path
from dataclasses import replace
from typing import Any

import numpy as np

from pymrst.plotting.boundary_faces import boundary_faces
from pymrst.plotting.private.plot_face_outline import plot_face_outline
from pymrst.plotting.private.plot_patches import (
    build_patch_geometry,
    plot_patches,
)
from pymrst.plotting.private.coarse_faces import (
    coarse_grid_parts,
    expand_coarse_faces,
)
from pymrst.plotting.results import PlotResult
from pymrst.plotting.private.rendering_compat import (
    configure_reservoir_camera,
    scaled_patch_geometry,
    translate_matlab_properties,
)


def _default_faces(g) -> np.ndarray:
    if g.griddim == 2:
        return np.arange(g.faces.num, dtype=np.int64)
    return boundary_faces(g)


def _expand_coarse_color(color, selected_count: int, groups):
    if isinstance(color, str) or color is None:
        return color
    values = np.asarray(color)
    if values.ndim == 1 and values.size != 3:
        values = values[:, None]
    if values.ndim == 2 and values.shape[0] == selected_count:
        return values[groups]
    return color


def plot_faces(
    g,
    faces=None,
    color: Any = "yellow",
    *,
    outline: bool = False,
    face_color=None,
    edge_color="black",
    opacity: float = 1.0,
    line_width: float = 1.0,
    flip_z: bool = True,
    zscale: float = 1.0,
    cmap: str = "viridis",
    clim=None,
    plotter=None,
    filename: str | Path | None = None,
    show: bool = True,
    **mesh_kwargs,
) -> PlotResult:
    """Plot a numeric/logical selection of 2-D or 3-D grid faces.

    Indices are Python/zero-based. ``faces=None`` follows MRST: every grid
    face in 2-D and the external boundary in 3-D.
    """
    parts = coarse_grid_parts(g)
    entity_grid = parts[0] if parts is not None else g
    if entity_grid.griddim not in (2, 3):
        raise ValueError("plot_faces supports only 2-D and 3-D grids")
    selected = _default_faces(entity_grid) if faces is None else faces
    # Normalize/validate against the grid the caller supplied.
    selected = build_patch_geometry(entity_grid, selected).entity_ids
    render_grid, render_faces, groups = expand_coarse_faces(g, selected)
    geometry = build_patch_geometry(render_grid, render_faces)
    geometry = scaled_patch_geometry(geometry, render_grid, zscale)
    if parts is not None:
        geometry = replace(geometry, group_ids=groups)

    render_color = (
        _expand_coarse_color(color, selected.size, groups)
        if parts is not None else color
    )

    try:
        import pyvista as pv
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "plot_faces requires PyVista; install pymrst[plotting]"
        ) from error

    own_plotter = plotter is None
    target = plotter or pv.Plotter(off_screen=filename is not None)
    translated = translate_matlab_properties(mesh_kwargs)
    if "color" in translated:
        face_color = translated.pop("color")
    if "edge_color" in translated:
        edge_color = translated.pop("edge_color")
    if "opacity" in translated:
        opacity = translated.pop("opacity")
    if "line_width" in translated:
        line_width = translated.pop("line_width")
    effective_color = (
        render_color if face_color is None else (
            _expand_coarse_color(face_color, selected.size, groups)
            if parts is not None else face_color
        )
    )
    no_fill = isinstance(effective_color, str) and (
        effective_color.lower() == "none"
    )

    result = PlotResult(geometry=geometry, plotter=target)
    if geometry.n_entities:
        kwargs = dict(translated)
        kwargs.setdefault("opacity", float(opacity))
        kwargs.setdefault("line_width", float(line_width))
        patch_color = effective_color
        if no_fill:
            # MATLAB PATCH with FaceColor='none' retains every polygon edge.
            # This is a wireframe, distinct from `outline=True`, which removes
            # edges internal to the whole selected face collection.
            kwargs.setdefault("style", "wireframe")
            patch_color = edge_color or "black"
        else:
            kwargs.setdefault("cmap", cmap)
            if clim is not None:
                kwargs.setdefault("clim", clim)
        if render_grid.griddim == 3 and not no_fill:
            kwargs.setdefault("show_edges", edge_color is not None)
            if parts is not None:
                kwargs["show_edges"] = False
            if edge_color is not None:
                kwargs.setdefault("edge_color", edge_color)
        rendered = plot_patches(
            render_grid, geometry.entity_ids, patch_color,
            plotter=target, show=False, zscale=zscale, **kwargs,
        )
        result.actor = rendered.actor

    if geometry.n_entities and (outline or parts is not None):
        outlined = plot_face_outline(
            render_grid,
            geometry.entity_ids,
            plotter=target,
            color=edge_color or "black",
            line_width=line_width,
            groups=groups if parts is not None else None,
            show=False,
        )
        result.outline_actor = outlined.actor

    configure_reservoir_camera(target, flip_z=flip_z, reset=own_plotter)
    if filename is not None:
        target.screenshot(str(filename))
        if own_plotter:
            target.close()
    elif show and own_plotter:
        target.show()
    return result


plotFaces = plot_faces

__all__ = ["plot_faces", "plotFaces"]
