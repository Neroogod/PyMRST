"""Plot cell data (MRST ``plotting/plotCellData.m``)."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from pymrst.plotting.boundary_faces import boundary_faces
from pymrst.plotting.plot_faces import plot_faces
from pymrst.plotting.private.coarse_faces import coarse_grid_parts
from pymrst.plotting.results import PatchGeometry, PlotResult
from pymrst.utils.get_sorted_cell_nodes import get_sorted_cell_nodes
from pymrst.plotting.private.rendering_compat import (
    configure_reservoir_camera,
    translate_matlab_properties,
)


def _cell_indices(g, cells) -> np.ndarray:
    if cells is None:
        return np.arange(g.cells.num, dtype=np.int64)
    raw = np.asarray(cells)
    if raw.dtype == bool:
        if raw.ndim != 1 or raw.size != g.cells.num:
            raise ValueError(
                f"logical cell mask has {raw.size} entries; "
                f"expected {g.cells.num}"
            )
        return np.flatnonzero(raw)
    if raw.size == 0:
        return np.empty(0, dtype=np.int64)
    numeric = np.asarray(raw, dtype=float).ravel()
    if not np.all(np.isfinite(numeric)):
        raise ValueError("cell indices must be finite")
    if not np.all(numeric == np.floor(numeric)):
        raise ValueError("cell indices must be integers")
    selected = numeric.astype(np.int64)
    if selected.min() < 0 or selected.max() >= g.cells.num:
        raise IndexError(f"cell indices must lie in [0, {g.cells.num-1}]")
    return selected


def _data_for_selection(g, data: Any, cells: np.ndarray) -> np.ndarray:
    values = np.asarray(data)
    if values.ndim == 0:
        values = values.reshape(1, 1)
    elif values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[1] not in (1, 3):
        raise ValueError("cell data must have one scalar or three RGB columns")
    if values.shape[0] == g.cells.num:
        return values[cells]
    if values.shape[0] == cells.size:
        return values
    raise ValueError(
        "cell data rows must equal the number of grid cells "
        f"({g.cells.num}) or selected cells ({cells.size}); "
        f"got {values.shape[0]}"
    )


def _cell_patch_geometry(g, cells: np.ndarray) -> PatchGeometry:
    all_nodes = get_sorted_cell_nodes(g)
    loops = tuple(
        all_nodes[g.cells.facePos[cell]:g.cells.facePos[cell+1]].copy()
        for cell in cells
    )
    if not loops:
        return PatchGeometry(
            np.empty((0, g.nodes.coords.shape[1])),
            (),
            cells.copy(),
            np.empty(0, dtype=np.int64),
            2,
        )
    node_ids = np.unique(np.concatenate(loops))
    local = np.full(g.nodes.num, -1, dtype=np.int64)
    local[node_ids] = np.arange(node_ids.size)
    return PatchGeometry(
        np.asarray(g.nodes.coords, dtype=float)[node_ids].copy(),
        tuple(local[loop] for loop in loops),
        cells.copy(),
        node_ids,
        2,
    )


def _plot_2d(
    g,
    cells: np.ndarray,
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
    geometry = _cell_patch_geometry(g, cells)
    try:
        import pyvista as pv
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "plot_cell_data requires PyVista; install pymrst[plotting]"
        ) from error
    own_plotter = plotter is None
    target = plotter or pv.Plotter(off_screen=filename is not None)
    result = PlotResult(geometry=geometry, plotter=target)
    if geometry.n_entities:
        points = geometry.vertices
        if points.shape[1] == 2:
            points = np.column_stack([points, np.zeros(points.shape[0])])
        stream = np.concatenate([
            np.r_[loop.size, loop] for loop in geometry.connectivity
        ]).astype(np.int64)
        mesh = pv.PolyData(points, faces=stream)
        mesh.field_data["grid_cell_ids"] = cells
        mesh.cell_data["cell_data"] = (
            values[:, 0] if values.shape[1] == 1 else values
        )
        kwargs = translate_matlab_properties(mesh_kwargs)
        kwargs.setdefault("scalars", "cell_data")
        kwargs.setdefault("opacity", float(opacity))
        kwargs.setdefault("show_edges", edge_color is not None)
        if edge_color is not None:
            kwargs.setdefault("edge_color", edge_color)
        if values.shape[1] == 3:
            kwargs.setdefault("rgb", True)
        else:
            kwargs.setdefault("cmap", cmap)
            if clim is not None:
                kwargs.setdefault("clim", clim)
        result.actor = target.add_mesh(mesh, **kwargs)
    configure_reservoir_camera(target, flip_z=True, reset=own_plotter)
    if filename is not None:
        target.screenshot(str(filename))
        if own_plotter:
            target.close()
    elif show and own_plotter:
        target.show()
    return result


def _plot_1d_series(
    g,
    cells: np.ndarray,
    values: np.ndarray,
    *,
    plotter,
    filename,
    show,
    opacity,
    mesh_kwargs,
) -> PlotResult:
    """Render MRST's ``plot(cell_centroids, values)`` one-dimensional path."""
    x = np.asarray(g.cells.centroids, dtype=float)[cells, 0]
    series = np.asarray(values, dtype=float)
    if series.ndim == 1:
        series = series[:, None]
    vertices = np.concatenate([
        np.column_stack([x, series[:, column]])
        for column in range(series.shape[1])
    ], axis=0) if cells.size else np.empty((0, 2), dtype=float)
    groups = (
        np.repeat(np.arange(series.shape[1]), cells.size)
        if series.shape[1] > 1 else None
    )
    geometry = PatchGeometry(
        vertices=vertices,
        connectivity=tuple(
            np.arange(column*cells.size, (column+1)*cells.size)
            for column in range(series.shape[1]) if cells.size
        ),
        entity_ids=np.tile(cells, series.shape[1]),
        node_ids=np.empty(0, dtype=np.int64),
        dimension=1,
        group_ids=groups,
    )
    try:
        import pyvista as pv
    except ImportError as error:  # pragma: no cover
        raise ImportError(
            "1-D plotting requires PyVista; install pymrst[plotting]"
        ) from error
    own_plotter = plotter is None
    target = plotter or pv.Plotter(off_screen=filename is not None)
    result = PlotResult(geometry=geometry, plotter=target)
    if cells.size:
        points = np.column_stack([
            vertices, np.zeros(vertices.shape[0], dtype=float)
        ])
        lines = []
        verts = []
        for connection in geometry.connectivity:
            if connection.size == 1:
                verts.extend([1, int(connection[0])])
            else:
                lines.extend([connection.size, *map(int, connection)])
        mesh = pv.PolyData(
            points,
            lines=np.asarray(lines, dtype=np.int64) if lines else None,
            verts=np.asarray(verts, dtype=np.int64) if verts else None,
        )
        mesh.field_data["grid_cell_ids"] = geometry.entity_ids
        kwargs = translate_matlab_properties(mesh_kwargs)
        kwargs.setdefault("opacity", float(opacity))
        result.actor = target.add_mesh(mesh, **kwargs)
    if filename is not None:
        target.screenshot(str(filename))
        if own_plotter:
            target.close()
    elif show and own_plotter:
        target.show()
    return result


def plot_cell_data(
    g,
    data: Any,
    cells=None,
    *,
    plotter=None,
    filename: str | Path | None = None,
    show: bool = True,
    cmap: str = "viridis",
    clim=None,
    edge_color=None,
    show_edges: bool | None = None,
    label: str = "",
    opacity: float = 1.0,
    flip_z: bool = True,
    zscale: float = 1.0,
    **mesh_kwargs,
) -> PlotResult:
    """Plot scalar or RGB cell data on a grid or cell subset."""
    if show_edges is not None:
        edge_color = "black" if show_edges else None
    if label:
        mesh_kwargs.setdefault("scalar_bar_args", {"title": label})
    parts = coarse_grid_parts(g)
    entity_grid = parts[0] if parts is not None else g
    selected = _cell_indices(entity_grid, cells)
    selected_values = _data_for_selection(entity_grid, data, selected)
    finite = ~np.any(np.isnan(selected_values.astype(float)), axis=1)
    selected = selected[finite]
    selected_values = selected_values[finite]

    if entity_grid.griddim == 3:
        faces, owners = boundary_faces(
            entity_grid, selected, with_cells=True
        )
        lookup = np.full(entity_grid.cells.num, -1, dtype=np.int64)
        lookup[selected] = np.arange(selected.size)
        face_values = selected_values[lookup[owners]]
        return plot_faces(
            g,
            faces,
            face_values,
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

    if entity_grid.griddim == 2:
        render_grid = entity_grid
        render_cells = selected
        render_values = selected_values
        if parts is not None:
            _, parent, partition = parts
            blocks = {
                cell: np.flatnonzero(partition == cell) for cell in selected
            }
            render_cells = np.concatenate(list(blocks.values())) if blocks else (
                np.empty(0, dtype=np.int64)
            )
            ordinal = {
                int(cell): i for i, cell in enumerate(selected)
            }
            render_values = np.asarray([
                selected_values[ordinal[int(partition[cell])]]
                for cell in render_cells
            ])
            render_grid = parent
        result = _plot_2d(
            render_grid,
            render_cells,
            render_values,
            plotter=plotter,
            filename=filename,
            show=show,
            cmap=cmap,
            clim=clim,
            edge_color=edge_color,
            opacity=opacity,
            mesh_kwargs=mesh_kwargs,
        )
        if parts is not None:
            groups = np.asarray([
                np.flatnonzero(selected == parts[2][cell])[0]
                for cell in render_cells
            ], dtype=np.int64)
            result.geometry = replace(result.geometry, group_ids=groups)
        return result

    if entity_grid.griddim == 1:
        if parts is not None:
            raise NotImplementedError(
                "coarse 1-D plotCellData is not represented by PyMRST grids"
            )
        return _plot_1d_series(
            entity_grid,
            selected,
            selected_values,
            plotter=plotter,
            filename=filename,
            show=show,
            opacity=opacity,
            mesh_kwargs=mesh_kwargs,
        )
    raise ValueError(f"unsupported grid dimension {entity_grid.griddim}")


plotCellData = plot_cell_data

__all__ = ["plot_cell_data", "plotCellData"]
