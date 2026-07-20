"""Browser-based interactive reservoir plotting.

This is PyMRST's lightweight answer to browser-native reservoir viewers such
as WGLMakie: prepare a grid/state/well scene in Python, then write a standalone
Plotly/WebGL HTML file that any browser can rotate, zoom, and inspect.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import webbrowser
from typing import Mapping, Sequence

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.params.wells_and_bc import Well
from pymrst.plotting.plot_toolbar import normalize_data


@dataclass
class BrowserReservoirPayload:
    """Geometry and fields used by the browser renderer."""

    centroids: np.ndarray
    fields: dict[str, np.ndarray]
    grid_lines: np.ndarray
    mesh_points: np.ndarray
    mesh_i: np.ndarray
    mesh_j: np.ndarray
    mesh_k: np.ndarray
    mesh_cell_index: np.ndarray
    wells: list[dict[str, np.ndarray | str | int]]
    times: np.ndarray | None = None


def reservoir_browser_data(
    g: Grid,
    data,
    *,
    wells: Sequence[Well] | Well | None = None,
    times: Sequence[float] | None = None,
    zscale: float = 1.0,
) -> BrowserReservoirPayload:
    """Prepare grid, fields, grid-lines, and wells for browser plotting."""
    centroids = _scaled_points(g.cells.centroids, zscale)
    fields = normalize_data(data, g.cells.num)
    grid_lines = _grid_line_points(g, zscale)
    return BrowserReservoirPayload(
        centroids=centroids,
        fields=fields,
        grid_lines=grid_lines,
        **_cell_mesh_payload(g, zscale),
        wells=_well_payloads(g, wells, zscale),
        times=None if times is None else np.asarray(times, dtype=float).ravel(),
    )


def plot_reservoir_browser(
    g: Grid,
    data,
    *,
    wells: Sequence[Well] | Well | None = None,
    times: Sequence[float] | None = None,
    filename: str | Path = "pymrst_reservoir_browser.html",
    title: str = "PyMRST reservoir browser",
    zscale: float = 1.0,
    marker_size: float = 4.0,
    cmap: str = "Viridis",
    style: str = "mesh",
    show_grid: bool = True,
    include_plotlyjs: str | bool = "cdn",
    open_browser: bool = False,
) -> Path:
    """Write an interactive browser reservoir viewer to an HTML file.

    The default viewer renders a triangulated cell-body mesh, not just points.
    The slider switches timestep/field frames. Grid lines and wells are ordinary
    Plotly legend traces, so the browser gives toggles without a server.
    """
    try:
        import plotly.graph_objects as go
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Browser plotting requires Plotly. Install with "
            "`pip install -e '.[browser]'` from the PyMRST root, or install "
            "`plotly` directly."
        ) from exc

    payload = reservoir_browser_data(g, data, wells=wells, times=times, zscale=zscale)
    traces = []
    names = list(payload.fields)
    if not names:
        raise ValueError("no fields supplied for browser plotting")
    first_name = names[0]
    first_series = payload.fields[first_name]
    if style not in {"mesh", "points"}:
        raise ValueError("style must be 'mesh' or 'points'")

    first_values = first_series[0]
    first_label = _field_step_label(first_name, 0, payload.times)
    if style == "mesh":
        if payload.mesh_points.size == 0:
            raise ValueError("grid did not produce a browser mesh")
        traces.append(
            go.Mesh3d(
                x=payload.mesh_points[:, 0],
                y=payload.mesh_points[:, 1],
                z=payload.mesh_points[:, 2],
                i=payload.mesh_i,
                j=payload.mesh_j,
                k=payload.mesh_k,
                intensity=first_values[payload.mesh_cell_index],
                intensitymode="vertex",
                colorscale=cmap,
                flatshading=True,
                opacity=0.92,
                name=first_label,
                colorbar={"title": first_name},
                showscale=True,
            )
        )
    else:
        traces.append(
            go.Scatter3d(
                x=payload.centroids[:, 0],
                y=payload.centroids[:, 1],
                z=payload.centroids[:, 2],
                mode="markers",
                marker={
                    "size": marker_size,
                    "color": first_values,
                    "colorscale": cmap,
                    "colorbar": {"title": first_name},
                },
                name=first_label,
                showlegend=False,
            )
        )

    frames = []
    for field_name in names:
        series = payload.fields[field_name]
        for step in range(series.shape[0]):
            label = _field_step_label(field_name, step, payload.times)
            if style == "mesh":
                frame_trace = go.Mesh3d(
                    intensity=series[step][payload.mesh_cell_index],
                    name=label,
                    colorbar={"title": field_name},
                )
            else:
                frame_trace = go.Scatter3d(
                    marker={"color": series[step], "colorbar": {"title": field_name}},
                    name=label,
                )
            frames.append(
                go.Frame(
                    data=[frame_trace],
                    traces=[0],
                    name=label,
                    layout={"title": f"{title}: {label}"},
                )
            )

    if show_grid and payload.grid_lines.size:
        traces.append(
            go.Scatter3d(
                x=payload.grid_lines[:, 0],
                y=payload.grid_lines[:, 1],
                z=payload.grid_lines[:, 2],
                mode="lines",
                line={"color": "rgba(35,35,35,0.28)", "width": 1},
                name="grid lines",
                visible=True,
            )
        )
    for well in payload.wells:
        pts = well["points"]
        color = "royalblue" if int(well["sign"]) > 0 else "crimson"
        traces.append(
            go.Scatter3d(
                x=pts[:, 0],
                y=pts[:, 1],
                z=pts[:, 2],
                mode="lines+markers",
                line={"color": color, "width": 8},
                marker={"size": marker_size + 1, "color": color},
                name=str(well["name"]),
                visible=True,
            )
        )

    fig = go.Figure(data=traces, frames=frames)
    slider_steps = [
        {
            "label": frame.name,
            "method": "animate",
            "args": [
                [frame.name],
                {
                    "mode": "immediate",
                    "frame": {"duration": 0, "redraw": True},
                    "transition": {"duration": 0},
                },
            ],
        }
        for frame in frames
    ]
    fig.update_layout(
        title=f"{title}: {first_label}",
        scene={
            "xaxis_title": "x",
            "yaxis_title": "y",
            "zaxis_title": "z",
            "aspectmode": "data",
        },
        legend={"itemsizing": "constant"},
        sliders=[
            {
                "active": 0,
                "currentvalue": {"prefix": "State: "},
                "pad": {"t": 36},
                "steps": slider_steps,
            }
        ] if slider_steps else [],
        updatemenus=[
            {
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [
                            None,
                            {
                                "frame": {"duration": 250, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "mode": "immediate",
                                "frame": {"duration": 0, "redraw": False},
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
                "direction": "left",
                "x": 0.02,
                "xanchor": "left",
                "y": 1.08,
                "yanchor": "top",
            }
        ] if frames else [],
        margin={"l": 0, "r": 0, "b": 0, "t": 68},
    )

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs=include_plotlyjs, full_html=True)
    if open_browser:
        webbrowser.open(path.resolve().as_uri())
    return path


def plot_saved_states_browser(
    state_file: str | Path,
    *,
    field_data: Mapping[str, np.ndarray] | None = None,
    filename: str | Path | None = None,
    deck_path: str | Path | None = None,
    zscale: float = 1.0,
    style: str = "mesh",
    field: str | None = None,
    all_fields: bool = False,
    open_browser: bool = False,
) -> Path:
    """Open a saved ``.npz`` state file in the browser viewer."""
    from pymrst.plotting.plot_saved_states import (
        build_spe1_plot_context,
        grid_from_saved_metadata,
        load_saved_state_fields,
        saved_cell_fields,
        wells_from_saved_metadata,
    )

    saved = load_saved_state_fields(state_file)
    if "grid.cells.num" in saved:
        g = grid_from_saved_metadata(saved)
        wells = wells_from_saved_metadata(saved)
    else:
        g, _, wells = build_spe1_plot_context(deck_path)
    if field_data is not None:
        fields = dict(field_data)
    else:
        fields = saved_cell_fields(saved)
        if field is not None and not all_fields:
            if field not in fields:
                raise KeyError(f"{field!r} not found in saved cell fields")
            fields = {field: fields[field]}
    out = filename or Path(state_file).with_suffix(".browser.html")
    return plot_reservoir_browser(
        g,
        fields,
        wells=wells,
        times=saved.get("time"),
        filename=out,
        title=f"PyMRST saved states: {Path(state_file).name}",
        zscale=zscale,
        style=style,
        open_browser=open_browser,
    )


def _field_step_label(field: str, step: int, times: np.ndarray | None) -> str:
    if times is None or step >= times.size:
        return f"{field} step {step}"
    return f"{field} step {step} ({times[step]:.6g} s)"


def _scaled_points(points: np.ndarray, zscale: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float).copy()
    if pts.shape[1] == 2:
        pts = np.column_stack([pts, np.zeros(pts.shape[0])])
    if not np.isclose(float(zscale), 1.0):
        z0 = float(np.min(pts[:, 2]))
        pts[:, 2] = z0 + (pts[:, 2] - z0) * float(zscale)
    return pts


def _grid_line_points(g: Grid, zscale: float) -> np.ndarray:
    coords = _scaled_points(g.nodes.coords, zscale)
    lines = []
    for f in range(g.faces.num):
        nodes = g.faces.nodes[g.faces.nodePos[f]:g.faces.nodePos[f + 1]]
        if nodes.size < 2:
            continue
        ring = list(map(int, nodes))
        if ring[0] != ring[-1]:
            ring.append(ring[0])
        for a, b in zip(ring[:-1], ring[1:]):
            lines.append(coords[a])
            lines.append(coords[b])
            lines.append([np.nan, np.nan, np.nan])
    return np.asarray(lines, dtype=float) if lines else np.zeros((0, 3), dtype=float)


def _cell_mesh_payload(g: Grid, zscale: float) -> dict[str, np.ndarray]:
    coords = _scaled_points(g.nodes.coords, zscale)
    points = []
    tri_i = []
    tri_j = []
    tri_k = []
    cell_index = []
    for c in range(g.cells.num):
        faces = g.cells.faces[g.cells.facePos[c]:g.cells.facePos[c + 1]]
        for f in faces:
            ring = list(map(int, g.faces.nodes[g.faces.nodePos[f]:g.faces.nodePos[f + 1]]))
            if len(ring) < 3:
                continue
            for pos in range(1, len(ring) - 1):
                base = len(points)
                points.extend([coords[ring[0]], coords[ring[pos]], coords[ring[pos + 1]]])
                tri_i.append(base)
                tri_j.append(base + 1)
                tri_k.append(base + 2)
                cell_index.extend([c, c, c])
    return {
        "mesh_points": np.asarray(points, dtype=float) if points else np.zeros((0, 3), dtype=float),
        "mesh_i": np.asarray(tri_i, dtype=np.int64),
        "mesh_j": np.asarray(tri_j, dtype=np.int64),
        "mesh_k": np.asarray(tri_k, dtype=np.int64),
        "mesh_cell_index": np.asarray(cell_index, dtype=np.int64),
    }


def _well_payloads(
    g: Grid,
    wells: Sequence[Well] | Well | None,
    zscale: float,
) -> list[dict[str, np.ndarray | str | int]]:
    if wells is None:
        return []
    if isinstance(wells, Well):
        wells = [wells]
    out = []
    for i, w in enumerate(wells):
        pts = _scaled_points(g.cells.centroids[np.asarray(w.cells, dtype=int)], zscale)
        if pts.shape[0] > 1:
            order = np.argsort(pts[:, 2])
            pts = pts[order]
        out.append({
            "name": getattr(w, "name", f"W{i + 1}"),
            "sign": int(getattr(w, "sign", 0)),
            "points": pts,
        })
    return out


__all__ = [
    "BrowserReservoirPayload",
    "plot_reservoir_browser",
    "plot_saved_states_browser",
    "reservoir_browser_data",
]
