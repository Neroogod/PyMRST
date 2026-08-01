"""Patch topology and rendering (MRST ``plotting/private/plotPatches.m``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from pymrst.plotting.results import ColorData, PatchGeometry, PlotResult
from pymrst.plotting.private.rendering_compat import (
    scaled_patch_geometry,
    translate_matlab_properties,
)


_NAMED_COLORS = {
    "y": (1.0, 1.0, 0.0),
    "yellow": (1.0, 1.0, 0.0),
    "m": (1.0, 0.0, 1.0),
    "magenta": (1.0, 0.0, 1.0),
    "c": (0.0, 1.0, 1.0),
    "cyan": (0.0, 1.0, 1.0),
    "r": (1.0, 0.0, 0.0),
    "red": (1.0, 0.0, 0.0),
    "g": (0.0, 1.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "b": (0.0, 0.0, 1.0),
    "blue": (0.0, 0.0, 1.0),
    "w": (1.0, 1.0, 1.0),
    "white": (1.0, 1.0, 1.0),
    "k": (0.0, 0.0, 0.0),
    "black": (0.0, 0.0, 0.0),
}


def _face_indices(g, faces) -> np.ndarray:
    raw = np.asarray(faces)
    if raw.dtype == bool:
        if raw.ndim != 1 or raw.size != g.faces.num:
            raise ValueError(
                f"logical face mask has {raw.size} entries; expected "
                f"{g.faces.num}"
            )
        selected = np.flatnonzero(raw)
    else:
        if raw.size == 0:
            return np.empty(0, dtype=np.int64)
        if not np.issubdtype(raw.dtype, np.number):
            raise TypeError("faces must be numeric indices or a logical mask")
        numeric = np.asarray(raw, dtype=float).ravel()
        if not np.all(np.isfinite(numeric)):
            raise ValueError("face indices must be finite")
        if not np.all(numeric == np.floor(numeric)):
            raise ValueError("face indices must be integers")
        selected = numeric.astype(np.int64)
    if selected.size and (
        selected.min() < 0 or selected.max() >= g.faces.num
    ):
        raise IndexError(
            f"face indices must lie in [0, {g.faces.num-1}]"
        )
    return selected


def build_patch_geometry(g, faces) -> PatchGeometry:
    """Return compact local topology for selected grid faces."""
    selected = _face_indices(g, faces)
    coords = np.asarray(g.nodes.coords, dtype=float)
    if selected.size == 0:
        return PatchGeometry(
            vertices=np.empty((0, coords.shape[1]), dtype=float),
            connectivity=(),
            entity_ids=selected,
            node_ids=np.empty(0, dtype=np.int64),
            dimension=int(g.griddim),
        )

    global_loops = tuple(
        np.asarray(
            g.faces.nodes[
                g.faces.nodePos[face]:g.faces.nodePos[face+1]
            ],
            dtype=np.int64,
        )
        for face in selected
    )
    if any(nodes.size < 2 for nodes in global_loops):
        raise ValueError("every plotted face must contain at least two nodes")
    node_ids = np.unique(np.concatenate(global_loops))
    local = np.full(g.nodes.num, -1, dtype=np.int64)
    local[node_ids] = np.arange(node_ids.size, dtype=np.int64)
    connectivity = tuple(local[nodes] for nodes in global_loops)
    return PatchGeometry(
        vertices=coords[node_ids].copy(),
        connectivity=connectivity,
        entity_ids=selected.copy(),
        node_ids=node_ids,
        dimension=int(g.griddim),
    )


def normalize_color_data(g, geometry: PatchGeometry, color: Any) -> ColorData:
    """Classify constant, per-face, and per-node color specifications."""
    if color is None:
        color = "yellow"
    if isinstance(color, str):
        return ColorData("constant", _NAMED_COLORS.get(color.lower(), color))

    values = np.asarray(color)
    if values.ndim == 0:
        return ColorData(
            "face",
            np.full((geometry.n_entities, 1), float(values)),
        )
    if values.ndim == 1 and values.size == 3:
        return ColorData("constant", values.astype(float))
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] not in (1, 3):
        raise ValueError("color data must have one or three columns")

    rows = values.shape[0]
    if rows == 1:
        if values.shape[1] == 3:
            return ColorData("constant", values[0].astype(float))
        return ColorData(
            "face",
            np.full((geometry.n_entities, 1), float(values[0, 0])),
        )
    if rows == geometry.n_entities:
        return ColorData("face", values.copy())
    if rows == g.nodes.num:
        return ColorData("node", values[geometry.node_ids].copy())
    raise ValueError(
        "color rows must be 1, the number of selected faces "
        f"({geometry.n_entities}), or the number of grid nodes "
        f"({g.nodes.num}); got {rows}"
    )


def _polydata(geometry: PatchGeometry):
    try:
        import pyvista as pv
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "plotting faces requires PyVista; install pymrst[plotting]"
        ) from error

    points = geometry.vertices
    if points.shape[1] == 2:
        points = np.column_stack([points, np.zeros(points.shape[0])])
    if geometry.dimension == 2:
        stream: list[int] = []
        for edge in geometry.connectivity:
            stream.extend((int(edge.size), *(int(v) for v in edge)))
        mesh = pv.PolyData(
            points, lines=np.asarray(stream, dtype=np.int64)
        )
    else:
        stream = []
        for polygon in geometry.connectivity:
            stream.extend((int(polygon.size), *(int(v) for v in polygon)))
        mesh = pv.PolyData(
            points, faces=np.asarray(stream, dtype=np.int64)
        )
    mesh.field_data["grid_face_ids"] = geometry.entity_ids
    return mesh


def plot_patches(
    g,
    faces,
    color: Any = "yellow",
    *,
    plotter=None,
    show: bool = False,
    zscale: float = 1.0,
    **mesh_kwargs,
) -> PlotResult:
    """Plot selected faces and return both actor and exact source geometry."""
    geometry = scaled_patch_geometry(build_patch_geometry(g, faces), g, zscale)
    colors = normalize_color_data(g, geometry, color)
    if geometry.n_entities == 0:
        return PlotResult(geometry=geometry, plotter=plotter)

    try:
        import pyvista as pv
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "plotting faces requires PyVista; install pymrst[plotting]"
        ) from error
    mesh = _polydata(geometry)
    kwargs = translate_matlab_properties(mesh_kwargs)
    if "color" in kwargs:
        color = kwargs.pop("color")
        colors = normalize_color_data(g, geometry, color)
    if colors.mode == "constant":
        kwargs.setdefault("color", colors.values)
    elif colors.mode == "face":
        name = "face_data"
        data = np.asarray(colors.values)
        mesh.cell_data[name] = data[:, 0] if data.shape[1] == 1 else data
        kwargs.setdefault("scalars", name)
        if data.shape[1] == 3:
            kwargs.setdefault("rgb", True)
    else:
        name = "node_data"
        data = np.asarray(colors.values)
        mesh.point_data[name] = data[:, 0] if data.shape[1] == 1 else data
        kwargs.setdefault("scalars", name)
        kwargs.setdefault("preference", "point")
        if data.shape[1] == 3:
            kwargs.setdefault("rgb", True)

    own_plotter = plotter is None
    target = plotter or pv.Plotter()
    actor = target.add_mesh(mesh, **kwargs)
    if show and own_plotter:
        target.show()
    return PlotResult(geometry=geometry, actor=actor, plotter=target)


plotPatches = plot_patches

__all__ = [
    "build_patch_geometry",
    "normalize_color_data",
    "plot_patches",
    "plotPatches",
]
