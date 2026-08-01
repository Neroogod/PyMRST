"""Browser-based interactive reservoir plotting.

This is PyMRST's lightweight answer to browser-native reservoir viewers such
as WGLMakie: prepare a grid/state/well scene in Python, then write a standalone
Plotly/WebGL HTML file that any browser can rotate, zoom, and inspect.

The viewer ships grid *topology* (nodes, face rings, face neighbours) rather
than a pre-triangulated solid, and builds the visible surface in the browser.
Only faces that separate a shown cell from a hidden one are drawn, so a filtered
reservoir renders its cut faces as solid geometry and the payload stays roughly
an order of magnitude smaller than a full per-cell triangulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
import html
import json
from pathlib import Path
import webbrowser
from typing import Mapping, Sequence

import numpy as np

from pymrst.gridprocessing.grid import Grid
from pymrst.params.wells_and_bc import Well
from pymrst.plotting.extensions.plot_toolbar import normalize_data


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
    nodes: np.ndarray = _dc_field(default_factory=lambda: np.zeros((0, 3)))
    face_pos: np.ndarray = _dc_field(default_factory=lambda: np.zeros(1, np.int64))
    face_nodes: np.ndarray = _dc_field(default_factory=lambda: np.zeros(0, np.int64))
    face_neighbors: np.ndarray = _dc_field(default_factory=lambda: np.zeros((0, 2), np.int64))
    ijk: np.ndarray | None = None
    cart_dims: tuple[int, ...] | None = None


def reservoir_browser_data(
    g: Grid,
    data,
    *,
    wells: Sequence[Well] | Well | None = None,
    times: Sequence[float] | None = None,
    zscale: float = 1.0,
    include_legacy_mesh: bool = True,
) -> BrowserReservoirPayload:
    """Prepare grid, fields, grid-lines, and wells for browser plotting.

    ``include_legacy_mesh`` builds the fully triangulated per-cell solid. The
    HTML writer does not need it (it triangulates the visible surface in the
    browser), so it passes ``False`` to avoid the memory cost on large grids.
    """
    centroids = _scaled_points(g.cells.centroids, zscale)
    fields = normalize_data(data, g.cells.num)
    grid_lines = _grid_line_points(g, zscale)
    mesh = _cell_mesh_payload(g, zscale) if include_legacy_mesh else _empty_mesh()
    return BrowserReservoirPayload(
        centroids=centroids,
        fields=fields,
        grid_lines=grid_lines,
        **mesh,
        wells=_well_payloads(g, wells, zscale),
        times=None if times is None else np.asarray(times, dtype=float).ravel(),
        **_topology(g),
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
    cmap: str | None = None,
    style: str = "mesh",
    show_grid: bool = True,
    include_plotlyjs: str | bool = "cdn",
    open_browser: bool = False,
) -> Path:
    """Write an interactive browser reservoir viewer to an HTML file.

    The viewer builds its visible surface client-side, so index filters act as
    true cutaways with solid cut faces. Colour limits default to the range over
    *all* timesteps, which keeps frames comparable during playback.
    """
    if style not in {"mesh", "points"}:
        raise ValueError("style must be 'mesh' or 'points'")

    payload = reservoir_browser_data(
        g, data, wells=wells, times=times, zscale=1.0, include_legacy_mesh=False
    )
    if not payload.fields:
        raise ValueError("no fields supplied for browser plotting")
    if style == "mesh" and payload.face_pos.size < 2:
        raise ValueError("grid did not produce a browser mesh")

    model = _scene_model(
        payload,
        style=style,
        cmap=cmap,
        zscale=zscale,
        marker_size=marker_size,
        show_grid=show_grid,
    )
    page = _engineering_page(model, title=title, include_plotlyjs=include_plotlyjs)

    path = Path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(page, encoding="utf-8")
    if open_browser:
        webbrowser.open(path.resolve().as_uri())
    return path


# --------------------------------------------------------------------------
# property semantics
# --------------------------------------------------------------------------

#: Scales that read backwards for a property family. Water saturation should
#: get *more* blue as water increases, which is Blues reversed.
_DEFAULT_REVERSE = {"saturation": True}

#: Colour map that reads best for each recognised property family. These are
#: viewer labels, not raw Plotly names -- Turbo is supplied as explicit stops
#: because Plotly.js has no built-in scale of that name and silently falls back
#: to RdBu when handed one it does not know.
_DEFAULT_CMAP = {
    "pressure": "Turbo",
    "saturation": "Blues",
    "permeability": "Viridis",
    "porosity": "Viridis",
    "temperature": "Hot",
    "generic": "Viridis",
}


def _field_kind(name: str) -> str:
    """Classify a field name so the viewer can offer sensible units."""
    n = str(name).strip().lower()
    if "pres" in n or n in {"p", "bhp", "po", "pw", "pg"}:
        return "pressure"
    if "sat" in n or (len(n) <= 3 and n.startswith("s")):
        return "saturation"
    if "perm" in n or n in {"k", "permx", "permy", "permz"}:
        return "permeability"
    if "poro" in n or n in {"phi", "poro"}:
        return "porosity"
    if "temp" in n or n in {"t"}:
        return "temperature"
    return "generic"


# --------------------------------------------------------------------------
# scene assembly
# --------------------------------------------------------------------------


def _scene_model(
    payload: BrowserReservoirPayload,
    *,
    style: str,
    cmap: str | None,
    zscale: float,
    marker_size: float,
    show_grid: bool,
) -> dict:
    """Build the JSON model the browser viewer runs on."""
    names = list(payload.fields)
    kinds = {name: _field_kind(name) for name in names}
    initial = cmap or _DEFAULT_CMAP[kinds[names[0]]]
    return {
        "nodes": _flat(payload.nodes),
        "facePos": [int(v) for v in payload.face_pos],
        "faceNodes": [int(v) for v in payload.face_nodes],
        "faceNeighbors": [int(v) for v in payload.face_neighbors.ravel()],
        "centroids": _flat(payload.centroids),
        "ijk": None if payload.ijk is None else [int(v) for v in payload.ijk.ravel()],
        "cartDims": None if payload.cart_dims is None else [int(v) for v in payload.cart_dims],
        "fields": {name: _json_array(series) for name, series in payload.fields.items()},
        "kinds": kinds,
        "defaultCmaps": {name: _DEFAULT_CMAP[kind] for name, kind in kinds.items()},
        "defaultReverse": {name: _DEFAULT_REVERSE.get(kind, False) for name, kind in kinds.items()},
        "times": [] if payload.times is None else [float(v) for v in payload.times],
        "wells": [
            {
                "name": str(w["name"]),
                "sign": int(w["sign"]),
                "points": _json_array(np.asarray(w["points"], dtype=float)),
            }
            for w in payload.wells
        ],
        "style": style,
        "initialScale": initial,
        "markerSize": float(marker_size),
        "showGrid": bool(show_grid),
        "zscale": float(zscale),
        "cellCount": int(payload.centroids.shape[0]),
    }


def _engineering_page(model: dict, *, title: str, include_plotlyjs: str | bool) -> str:
    """Wrap the viewer in a reservoir-engineering control workspace."""
    safe_title = html.escape(title)
    encoded = json.dumps(model, separators=(",", ":"), allow_nan=False)
    # Guard the JSON payload against prematurely closing its own script tag.
    encoded = encoded.replace("</", "<\\/")

    if include_plotlyjs is True:
        from plotly.offline import get_plotlyjs

        plotly_tag = f"<script>{get_plotlyjs()}</script>"
    elif include_plotlyjs == "cdn":
        from plotly.offline import get_plotlyjs_version

        plotly_tag = (
            f'<script src="https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"></script>'
        )
    else:
        plotly_tag = ""

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{safe_title}</title>
<style>{_PAGE_CSS}</style></head><body>
<main class="shell">
<aside class="rail" id="rail">
  <div class="brand"><div class="mark">P</div>
    <div><h1>PyMRST Studio</h1><div class="subtitle">Reservoir state explorer</div></div></div>

  <div class="section"><div class="section-title">Property</div>
    <select id="field"></select>
    <div class="pair"><div><label for="unit">Units</label><select id="unit"></select></div>
      <div><label for="cmap">Colour map</label><select id="cmap"></select></div></div>
    <label class="check"><input id="reverse" type="checkbox"> Reverse colour scale</label>
    <label class="check"><input id="delta" type="checkbox"> Change from initial state</label>
  </div>

  <div class="section"><div class="section-title">Colour range</div>
    <select id="range-mode">
      <option value="global">Global — all timesteps</option>
      <option value="step">This timestep only</option>
      <option value="manual">Manual</option>
    </select>
    <label class="check"><input id="robust" type="checkbox"> Clip outliers (P2–P98)</label>
    <div class="pair"><div><label for="cmin">Min</label><input id="cmin" type="number" step="any"></div>
      <div><label for="cmax">Max</label><input id="cmax" type="number" step="any"></div></div>
    <div class="help" id="range-note"></div>
  </div>

  <div class="section" id="filter-section"><div class="section-title">Index filter</div>
    <div class="filter-row"><span class="axis">I</span>
      <input id="i0" type="range"><input id="i1" type="range"><span class="rng" id="i-label"></span></div>
    <div class="filter-row"><span class="axis">J</span>
      <input id="j0" type="range"><input id="j1" type="range"><span class="rng" id="j-label"></span></div>
    <div class="filter-row"><span class="axis">K</span>
      <input id="k0" type="range"><input id="k1" type="range"><span class="rng" id="k-label"></span></div>
    <div class="button-row"><button id="filter-reset">Reset filter</button>
      <button id="filter-top">Top layer</button></div>
    <div class="help">Cut faces render solid, so the interior reads as rock, not a shell.</div>
  </div>

  <div class="section"><div class="section-title">Display</div>
    <label for="ve">Vertical exaggeration <span class="val" id="ve-value"></span></label>
    <input id="ve" type="range" min="1" max="40" step="1">
    <label for="opacity">Opacity <span class="val" id="opacity-value"></span></label>
    <input id="opacity" type="range" min="15" max="100">
    <label for="theme">Appearance</label>
    <select id="theme">
      <option value="auto">Auto — match system</option>
      <option value="light">Light</option>
      <option value="dark">Dark</option>
    </select>
    <label class="check"><input id="grid-toggle" type="checkbox"> Cell edges</label>
    <label class="check"><input id="well-toggle" type="checkbox"> Wells</label>
    <div class="button-row"><button data-camera="iso">Isometric</button><button data-camera="top">Map</button>
      <button data-camera="front">Front</button><button data-camera="side">Side</button></div>
  </div>
</aside>

<section class="viewer">
  <button class="panel-toggle rail-toggle" id="rail-toggle" aria-label="Toggle controls" aria-expanded="false">&#9776;</button>
  <button class="panel-toggle inspector-toggle" id="inspector-toggle" aria-label="Toggle statistics" aria-expanded="false">&#9878;</button>
  <div class="topbar"><div class="title-card"><strong>{safe_title}</strong>
    <span class="badge" id="state-badge"></span>
    <div class="help">Drag to orbit &middot; scroll to zoom &middot; click a cell to inspect &middot; &larr;/&rarr; step &middot; space play</div></div>
  </div>
  <div id="pymrst-reservoir-view"></div>
  <div class="transport">
    <button id="first" title="First step">&#9198;</button>
    <button id="previous" title="Previous step">&#9664;</button>
    <button id="play" class="primary" title="Play/pause">&#9654;</button>
    <button id="next" title="Next step">&#9654;</button>
    <button id="last" title="Last step">&#9197;</button>
    <input id="step" type="range" min="0" value="0" step="1">
    <select id="time-unit"></select>
    <span class="time-value" id="time-value"></span>
  </div>
</section>

<aside class="inspector" id="inspector">
  <div class="section"><div class="section-title">State statistics</div>
    <div class="stats">
      <div class="stat"><b id="stat-min">&ndash;</b><span>Minimum</span></div>
      <div class="stat"><b id="stat-max">&ndash;</b><span>Maximum</span></div>
      <div class="stat"><b id="stat-mean">&ndash;</b><span>Mean</span></div>
      <div class="stat"><b id="stat-p90">&ndash;</b><span>P90</span></div>
      <div class="stat"><b id="stat-shown">&ndash;</b><span>Cells shown</span></div>
      <div class="stat"><b id="stat-wells">&ndash;</b><span>Wells</span></div>
    </div></div>
  <div class="section"><div class="section-title">Distribution</div>
    <canvas id="hist" height="110"></canvas>
    <div class="help" id="hist-note">Shaded band is the active colour range.</div></div>
  <div class="section"><div class="section-title">History</div>
    <canvas id="series" height="120"></canvas>
    <div class="legend-row"><span class="key mean"></span>Field mean
      <span class="key cell"></span><span id="series-cell">no cell selected</span></div></div>
  <div class="section"><div class="section-title">Selection</div>
    <div id="selection" class="selection-body">Click a cell in the 3D view.</div></div>
</aside>
</main>
<script id="pymrst-model" type="application/json">{encoded}</script>
{plotly_tag}
<script>{_PAGE_JS}</script>
</body></html>"""


# --------------------------------------------------------------------------
# geometry helpers
# --------------------------------------------------------------------------


def _flat(points) -> list[float]:
    array = np.asarray(points, dtype=float)
    return [0.0 if not np.isfinite(v) else float(v) for v in array.ravel()]


def _json_array(values) -> list:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    return [
        [None if not np.isfinite(value) else float(value) for value in row]
        for row in array
    ]


def _empty_mesh() -> dict[str, np.ndarray]:
    return {
        "mesh_points": np.zeros((0, 3), dtype=float),
        "mesh_i": np.zeros(0, dtype=np.int64),
        "mesh_j": np.zeros(0, dtype=np.int64),
        "mesh_k": np.zeros(0, dtype=np.int64),
        "mesh_cell_index": np.zeros(0, dtype=np.int64),
    }


def _topology(g: Grid) -> dict:
    """Compact face topology: the browser triangulates from this."""
    coords = np.asarray(g.nodes.coords, dtype=float)
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(coords.shape[0])])
    return {
        "nodes": coords,
        "face_pos": np.asarray(g.faces.nodePos, dtype=np.int64),
        "face_nodes": np.asarray(g.faces.nodes, dtype=np.int64),
        "face_neighbors": np.asarray(g.faces.neighbors, dtype=np.int64).reshape(-1, 2),
        "ijk": _cell_ijk(g),
        "cart_dims": tuple(int(v) for v in g.cartDims) if getattr(g, "cartDims", None) is not None else None,
    }


def _cell_ijk(g: Grid) -> np.ndarray | None:
    """Logical (i, j, k) per active cell, when the grid is index-addressable."""
    dims = getattr(g, "cartDims", None)
    if dims is None or len(dims) < 3:
        return None
    nx, ny = int(dims[0]), int(dims[1])
    index_map = getattr(g.cells, "indexMap", None)
    idx = (
        np.arange(g.cells.num, dtype=np.int64)
        if index_map is None
        else np.asarray(index_map, dtype=np.int64).ravel()
    )
    if idx.size != g.cells.num:
        return None
    return np.column_stack([idx % nx, (idx // nx) % ny, idx // (nx * ny)]).astype(np.int64)


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
    from pymrst.plotting.extensions.plot_saved_states import (
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


_PAGE_CSS = """
/* Dark is the default; the light overrides below are switched on by a
   data-theme attribute that the viewer resolves (auto / light / dark). */
:root{--bg:#060b13;--panel:#0b1423;--panel2:#101d31;--line:#1e2d44;--text:#e2ecf9;
--muted:#7e91ab;--accent:#38bdf8;--accent-soft:#12304b;--hover:#14263e;
--overlay:#08111fd9;--rail-a:#0b1423;--rail-b:#070e1a;--shadow:0 8px 30px #0006;
--inj:#3b82f6;--prod:#ef4444}
:root[data-theme="light"]{--bg:#eef2f7;--panel:#ffffff;--panel2:#f4f7fb;--line:#d7e0ea;
--text:#0f1d2e;--muted:#5c6c81;--accent:#0369a1;--accent-soft:#dbeefc;--hover:#e4edf6;
--overlay:#ffffffe0;--rail-a:#ffffff;--rail-b:#f4f7fb;--shadow:0 8px 30px #0f1d2e1a;
--inj:#1d4ed8;--prod:#b91c1c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);overflow:hidden;
font:13px/1.45 Inter,ui-sans-serif,system-ui,-apple-system,sans-serif}
.shell{display:grid;grid-template-columns:296px 1fr 288px;height:100vh}
.rail,.inspector{background:linear-gradient(180deg,var(--rail-a),var(--rail-b));overflow:auto;z-index:4}
.rail{border-right:1px solid var(--line);padding:16px}
.inspector{border-left:1px solid var(--line);padding:16px}
.brand{display:flex;align-items:center;gap:11px;margin-bottom:6px}
.mark{width:32px;height:32px;border-radius:9px;background:linear-gradient(145deg,#38bdf8,#2563eb);
display:grid;place-items:center;font-weight:800;color:#fff;box-shadow:0 0 22px #38bdf855}
h1{font-size:15px;margin:0;letter-spacing:-.01em}
.subtitle{color:var(--muted);font-size:11px}
.section{padding:13px 0;border-top:1px solid var(--line)}
.section-title{font-size:10px;text-transform:uppercase;letter-spacing:.13em;color:var(--accent);margin-bottom:9px}
label{display:block;color:var(--muted);margin:8px 0 4px;font-size:11.5px}
label.check{display:flex;align-items:center;gap:7px;margin:9px 0 0;color:var(--text);cursor:pointer}
label.check input{accent-color:var(--accent);margin:0}
.val{color:var(--accent);font-variant-numeric:tabular-nums;float:right}
select,input[type=number]{width:100%;background:var(--panel2);color:var(--text);
border:1px solid var(--line);border-radius:7px;padding:7px 8px;font:inherit;font-size:12px}
select:focus,input:focus{outline:none;border-color:var(--accent)}
input[type=range]{width:100%;accent-color:var(--accent);margin:2px 0}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:8px}
.button-row{display:grid;grid-template-columns:repeat(2,1fr);gap:6px;margin-top:9px}
button{border:1px solid var(--line);background:var(--panel2);color:var(--text);border-radius:7px;
padding:7px;cursor:pointer;font:inherit;font-size:12px}
button:hover{background:var(--hover);border-color:var(--accent)}
.help{color:var(--muted);font-size:10.5px;margin-top:7px;line-height:1.4}
.filter-row{display:grid;grid-template-columns:14px 1fr 1fr 62px;gap:6px;align-items:center;margin:6px 0}
.axis{color:var(--accent);font-weight:700;font-size:11px}
.rng{color:var(--muted);font-size:10.5px;text-align:right;font-variant-numeric:tabular-nums}
.viewer{position:relative;min-width:0;display:flex;flex-direction:column}
#pymrst-reservoir-view{flex:1;min-height:0}
.topbar{position:absolute;left:16px;top:12px;z-index:3;pointer-events:none}
.title-card{background:var(--overlay);border:1px solid var(--line);border-radius:10px;padding:9px 12px;
backdrop-filter:blur(10px);box-shadow:var(--shadow);max-width:min(460px,52vw)}
.badge{display:inline-block;padding:2px 8px;border-radius:99px;background:var(--accent-soft);
color:var(--accent);font-size:10px;margin-left:6px;font-variant-numeric:tabular-nums}
.transport{display:flex;align-items:center;gap:8px;padding:9px 14px;border-top:1px solid var(--line);
background:var(--panel)}
.transport button{padding:5px 9px;min-width:34px}
.transport button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.transport #step{flex:1}
.transport select{width:auto;padding:5px 7px}
.time-value{color:var(--accent);font-variant-numeric:tabular-nums;min-width:118px;text-align:right;font-size:12px}
.stats{display:grid;grid-template-columns:repeat(2,1fr);gap:6px}
.stat{background:var(--panel2);border:1px solid var(--line);border-radius:8px;padding:8px}
.stat b{display:block;font-size:13.5px;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.stat span{font-size:9.5px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}
canvas{width:100%;display:block;border-radius:7px;background:var(--panel2);border:1px solid var(--line)}
.legend-row{display:flex;align-items:center;gap:6px;color:var(--muted);font-size:10.5px;margin-top:7px}
.key{width:14px;height:3px;border-radius:2px;display:inline-block}
.key.mean{background:var(--accent)}.key.cell{background:#f59e0b;margin-left:8px}
.selection-body{color:var(--muted);font-size:11.5px;line-height:1.6}
.selection-body b{color:var(--accent);font-size:13px}
.panel-toggle{display:none}
@media(max-width:1180px){.shell{grid-template-columns:296px 1fr}.inspector{position:absolute;right:0;top:0;
width:288px;height:100%;transform:translateX(100%);transition:.2s;border-left:1px solid var(--line);z-index:5}
.inspector.open{transform:none}
.inspector-toggle{display:grid;place-items:center;position:absolute;top:12px;right:12px;z-index:6;
width:36px;height:36px;padding:0;font-size:16px;background:var(--overlay);border:1px solid var(--line);
border-radius:9px;backdrop-filter:blur(10px)}
.topbar{right:58px}}
@media(max-width:820px){.shell{grid-template-columns:1fr}
.rail{position:absolute;width:288px;height:100%;transform:translateX(-100%);transition:.2s;z-index:5}
.rail.open{transform:none}
.rail-toggle{display:grid;place-items:center;position:absolute;top:12px;left:12px;z-index:6;
width:36px;height:36px;padding:0;font-size:16px;background:var(--overlay);border:1px solid var(--line);
border-radius:9px;backdrop-filter:blur(10px)}
.topbar{left:58px;right:58px}}
"""


_PAGE_JS = r"""
(function(){
'use strict';
var M = JSON.parse(document.getElementById('pymrst-model').textContent);
var $ = function(id){ return document.getElementById(id); };
var graph = $('pymrst-reservoir-view');
var NC = M.cellCount;
var NF = M.facePos.length - 1;
var NAMES = Object.keys(M.fields);

/* ---------------- units ---------------- */
/* [label, multiplier, offset]; display = raw*multiplier + offset */
var UNIT_SETS = {
  pressure:    [['Pa',1,0],['bar',1e-5,0],['MPa',1e-6,0],['psi',1.4503773773e-4,0]],
  saturation:  [['fraction',1,0],['%',100,0]],
  permeability:[['m2',1,0],['mD',1.01324997e15,0],['D',1.01324997e12,0]],
  porosity:    [['fraction',1,0],['%',100,0]],
  temperature: [['K',1,0],['degC',1,-273.15]],
  generic:     [['--',1,0]]
};
var DEFAULT_UNIT = {pressure:1, saturation:0, permeability:1, porosity:0, temperature:1, generic:0};
var TIME_UNITS = [['s',1],['hours',1/3600],['days',1/86400],['years',1/(86400*365.25)]];
/* Plotly.js only knows a fixed set of named scales and silently substitutes
   RdBu for anything else, so Turbo ships as explicit stops. */
var TURBO = [[0,'#30123b'],[0.1,'#4458cb'],[0.2,'#3e9bfe'],[0.3,'#18d6cb'],[0.4,'#46f884'],
             [0.5,'#a2fc3c'],[0.6,'#e2dc37'],[0.7,'#fea331'],[0.8,'#ef5a11'],[0.9,'#c02a04'],[1,'#7a0403']];
var CMAPS = [['Turbo',TURBO],['Viridis','Viridis'],['Cividis','Cividis'],['Jet','Jet'],
             ['Blues','Blues'],['YlGnBu','YlGnBu'],['RdBu','RdBu'],['Portland','Portland'],
             ['Earth','Earth'],['Hot','Hot'],['Greys','Greys']];
function scaleFor(label){
  for (var i=0;i<CMAPS.length;i++) if (CMAPS[i][0]===label) return CMAPS[i][1];
  return 'Viridis';
}

var BOUNDS = (function(){
  var lo=[Infinity,Infinity,Infinity], hi=[-Infinity,-Infinity,-Infinity];
  for (var i=0;i<M.nodes.length;i+=3){
    for (var d=0;d<3;d++){ var v=M.nodes[i+d]; if(v<lo[d])lo[d]=v; if(v>hi[d])hi[d]=v; }
  }
  return [Math.max(hi[0]-lo[0],1e-9), Math.max(hi[1]-lo[1],1e-9), Math.max(hi[2]-lo[2],1e-9)];
})();

/* Plotly and the 2D canvases paint their own pixels, so the theme cannot be
   expressed in CSS alone -- these mirror the stylesheet variables. */
var THEMES = {
  dark:  {paper:'#060b13', axisBg:'#0a1424', grid:'#20304a', zero:'#2b3f5c',
          font:'#dbeafe', edge:'rgba(148,163,184,0.18)', legendBg:'rgba(8,17,31,.72)',
          canvas:'#101d31', axisLine:'#1e2d44', bar:'#38bdf8', barDim:'#31445f',
          band:'rgba(56,189,248,0.10)', tick:'#7e91ab', marker:'#fbbf24', cursor:'rgba(226,236,249,0.45)'},
  light: {paper:'#eef2f7', axisBg:'#ffffff', grid:'#d7e0ea', zero:'#b9c6d6',
          font:'#17324d', edge:'rgba(71,85,105,0.30)', legendBg:'rgba(255,255,255,.85)',
          canvas:'#f4f7fb', axisLine:'#d7e0ea', bar:'#0284c7', barDim:'#c3d0de',
          band:'rgba(2,132,199,0.12)', tick:'#5c6c81', marker:'#b45309', cursor:'rgba(15,29,46,0.35)'}
};
function resolvedTheme(){
  if (S.theme === 'auto'){
    return (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches)
      ? 'light' : 'dark';
  }
  return S.theme;
}
function TH(){ return THEMES[resolvedTheme()]; }

var S = {
  field: NAMES[0], step: 0, unit: {}, timeUnit: M.times.length ? 2 : 0,
  rangeMode: 'global', robust: false, manual: {lo:null, hi:null},
  cmap: M.initialScale, reverse: false, opacity: 1.0, ve: M.zscale, theme: 'auto',
  showGrid: M.showGrid, showWells: true, delta: false,
  filt: null, selected: null, playing: false, timer: null
};
NAMES.forEach(function(n){ S.unit[n] = DEFAULT_UNIT[M.kinds[n]] || 0; });
S.reverse = !!(M.defaultReverse && M.defaultReverse[S.field]);

/* ---------------- numeric helpers ---------------- */
/* Looped, not spread: Math.min(...huge) throws RangeError past ~100k cells. */
function minMax(a){
  var lo = Infinity, hi = -Infinity;
  for (var i=0;i<a.length;i++){ var v=a[i];
    if (v===null || !isFinite(v)) continue;
    if (v<lo) lo=v; if (v>hi) hi=v; }
  return [lo,hi];
}
function finite(a){ var o=[]; for (var i=0;i<a.length;i++){ var v=a[i];
  if (v!==null && isFinite(v)) o.push(v); } return o; }
function quantile(sorted,q){
  if(!sorted.length) return NaN;
  var p=(sorted.length-1)*q, b=Math.floor(p), r=p-b;
  return sorted[b] + (sorted[Math.min(b+1,sorted.length-1)] - sorted[b])*r;
}
function fmt(v){
  if (v===null || !isFinite(v)) return '–';
  var a=Math.abs(v);
  if (a>=1e6 || (a>0 && a<1e-3)) return v.toExponential(3);
  return v.toLocaleString(undefined,{maximumSignificantDigits:5});
}

/* ---------------- values, units, ranges ---------------- */
function unitFor(name){ return UNIT_SETS[M.kinds[name]||'generic'][S.unit[name]||0]; }
function unitLabel(){ var u=unitFor(S.field); return u[0]==='--' ? '' : u[0]; }
function axisLabel(){ var u=unitLabel(); return S.field + (S.delta?' change':'') + (u?' ['+u+']':''); }

function convert(v){
  if (v===null || !isFinite(v)) return null;
  var u = unitFor(S.field);
  /* A difference carries the scale but not the zero-point offset. */
  return S.delta ? v*u[1] : v*u[1] + u[2];
}
function valuesAt(step){
  var series = M.fields[S.field];
  var k = Math.min(step, series.length-1);
  var raw = series[k], base = S.delta ? series[0] : null, out = new Array(NC);
  for (var c=0;c<NC;c++){
    var v = raw[c];
    if (v===null){ out[c]=null; continue; }
    if (base){ var b=base[c]; out[c] = (b===null) ? null : convert(v-b); }
    else out[c] = convert(v);
  }
  return out;
}

var rangeCache = {};
function globalRange(){
  var key = S.field+'|'+S.unit[S.field]+'|'+(S.delta?1:0)+'|'+(S.robust?1:0);
  if (rangeCache[key]) return rangeCache[key];
  var series = M.fields[S.field], all = [];
  for (var k=0;k<series.length;k++){
    var v = valuesAt(k);
    for (var c=0;c<NC;c++){ var x=v[c]; if (x!==null && isFinite(x)) all.push(x); }
  }
  var r;
  if (!all.length) r=[0,1];
  else if (S.robust){ all.sort(function(a,b){return a-b;}); r=[quantile(all,0.02), quantile(all,0.98)]; }
  else r = minMax(all);
  if (!(r[1]>r[0])) r=[r[0]-0.5, r[0]+0.5];
  rangeCache[key]=r; return r;
}
function stepRange(){
  var vals = finite(valuesAt(S.step));
  if (!vals.length) return [0,1];
  var r;
  if (S.robust){ vals.sort(function(a,b){return a-b;}); r=[quantile(vals,0.02), quantile(vals,0.98)]; }
  else r = minMax(vals);
  if (!(r[1]>r[0])) r=[r[0]-0.5, r[0]+0.5];
  return r;
}
/* A change map is only readable if zero sits at the centre of a diverging
   scale, so difference ranges are forced symmetric. */
function symmetric(r){
  var m = Math.max(Math.abs(r[0]), Math.abs(r[1]));
  return m>0 ? [-m, m] : [-1, 1];
}
function colorRange(){
  if (S.rangeMode==='manual' && S.manual.lo!==null && S.manual.hi!==null && S.manual.hi>S.manual.lo)
    return [S.manual.lo, S.manual.hi];
  var r = S.rangeMode==='step' ? stepRange() : globalRange();
  return S.delta ? symmetric(r) : r;
}

/* ---------------- visibility + surface ---------------- */
function computeVisible(){
  var vis = new Uint8Array(NC);
  if (!M.ijk || !S.filt){ vis.fill(1); return vis; }
  var f=S.filt;
  for (var c=0;c<NC;c++){
    var i=M.ijk[3*c], j=M.ijk[3*c+1], k=M.ijk[3*c+2];
    vis[c] = (i>=f.i[0]&&i<=f.i[1] && j>=f.j[0]&&j<=f.j[1] && k>=f.k[0]&&k<=f.k[1]) ? 1 : 0;
  }
  return vis;
}

var surface = null;
/* Draw a face only where it separates a shown cell from a hidden one: that is
   the reservoir's visible skin, and it makes filters read as solid cutaways. */
function buildSurface(){
  var vis = computeVisible();
  var N=M.nodes, FP=M.facePos, FN=M.faceNodes, NB=M.faceNeighbors;
  var X=[],Y=[],Z=[],I=[],J=[],K=[],owner=[],EX=[],EY=[],EZ=[];
  var NN=N.length/3, seenEdge=new Set();
  var shown=0; for (var c=0;c<NC;c++) if (vis[c]) shown++;
  for (var f=0;f<NF;f++){
    var a=NB[2*f], b=NB[2*f+1];
    var va = a>=0 && vis[a]===1, vb = b>=0 && vis[b]===1;
    if (va===vb) continue;
    var own = va ? a : b;
    var s=FP[f], e=FP[f+1], m=e-s;
    if (m<3) continue;
    var base=X.length;
    for (var p=s;p<e;p++){
      var n=FN[p];
      X.push(N[3*n]); Y.push(N[3*n+1]); Z.push(N[3*n+2]); owner.push(own);
    }
    for (var t=1;t<m-1;t++){ I.push(base); J.push(base+t); K.push(base+t+1); }
    for (var q=0;q<m;q++){
      /* Adjacent faces share an edge; emit each undirected node pair once. */
      var na=FN[s+q], nb=FN[s+((q+1)%m)];
      var key = na<nb ? na*NN+nb : nb*NN+na;
      if (seenEdge.has(key)) continue;
      seenEdge.add(key);
      EX.push(N[3*na],N[3*nb],null); EY.push(N[3*na+1],N[3*nb+1],null);
      EZ.push(N[3*na+2],N[3*nb+2],null);
    }
  }
  var meta = new Array(owner.length);
  for (var v2=0;v2<owner.length;v2++) meta[v2] = cellRecord(owner[v2]);
  var pidx = [], pmeta = [];
  if (M.style !== 'mesh'){
    for (var c2=0;c2<NC;c2++) if (vis[c2]){ pidx.push(c2); pmeta.push(cellRecord(c2)); }
  }
  surface = {X:X,Y:Y,Z:Z,I:I,J:J,K:K,owner:owner,meta:meta,
             pointIdx:pidx, pointMeta:pmeta,
             EX:EX,EY:EY,EZ:EZ,vis:vis,shown:shown};
}

/* ---------------- traces + layout ---------------- */
/* Per-vertex identity is [cell, i, j, k, x, y, depth]. It depends only on the
   filter and exaggeration, so it is cached with the surface rather than rebuilt
   each timestep; the changing value is read from the trace via valueRef. */
function cellRecord(c){
  var i='-', j='-', k='-';
  if (M.ijk){ i=M.ijk[3*c]; j=M.ijk[3*c+1]; k=M.ijk[3*c+2]; }
  return [c, i, j, k, M.centroids[3*c], M.centroids[3*c+1], M.centroids[3*c+2]];
}
function hoverTemplate(valueRef){
  /* Hover labels are SVG text, so HTML entities are not decoded -- use the
     literal separator, not &middot;. */
  var dot = ' · ';
  return '<b>Cell %{customdata[0]}</b><br>'
    + (M.ijk ? 'I %{customdata[1]}' + dot + 'J %{customdata[2]}' + dot + 'K %{customdata[3]}<br>' : '')
    + 'X %{customdata[4]:.6g}' + dot + 'Y %{customdata[5]:.6g}' + dot + 'Depth %{customdata[6]:.6g}<br>'
    + axisLabel() + ' <b>%{' + valueRef + ':.6g}</b><extra></extra>';
}
function buildTraces(){
  var vals = valuesAt(S.step), r = colorRange(), out = [];
  if (M.style==='mesh'){
    var inten=new Array(surface.owner.length);
    for (var v=0;v<surface.owner.length;v++) inten[v]=vals[surface.owner[v]];
    out.push({type:'mesh3d', x:surface.X, y:surface.Y, z:surface.Z,
      i:surface.I, j:surface.J, k:surface.K,
      intensity:inten, intensitymode:'vertex', colorscale:scaleFor(S.cmap),
      reversescale:S.reverse,
      cmin:r[0], cmax:r[1], opacity:S.opacity, flatshading:true, showscale:true,
      colorbar:{title:{text:axisLabel(), side:'right'}, len:0.62, y:0.5, yanchor:'middle',
                x:1.005, thickness:13, outlinewidth:0, tickfont:{size:10}},
      customdata:surface.meta, hovertemplate:hoverTemplate('intensity'), name:'reservoir', showlegend:false});
  } else {
    var px=[],py=[],pz=[],pc=[];
    for (var n=0;n<surface.pointIdx.length;n++){
      var c2=surface.pointIdx[n];
      px.push(M.centroids[3*c2]); py.push(M.centroids[3*c2+1]); pz.push(M.centroids[3*c2+2]);
      pc.push(vals[c2]);
    }
    var pd = surface.pointMeta;
    out.push({type:'scatter3d', mode:'markers', x:px, y:py, z:pz,
      marker:{size:M.markerSize, color:pc, colorscale:scaleFor(S.cmap), cmin:r[0], cmax:r[1],
        reversescale:S.reverse, opacity:S.opacity, showscale:true,
        colorbar:{title:{text:axisLabel(), side:'right'}, len:0.62, y:0.5, yanchor:'middle',
                  x:1.005, thickness:13, outlinewidth:0, tickfont:{size:10}}},
      customdata:pd, hovertemplate:hoverTemplate('marker.color'), name:'reservoir', showlegend:false});
  }
  edgeTrace = -1;
  if (S.showGrid && surface.EX.length){
    edgeTrace = out.length;
    out.push({type:'scatter3d', mode:'lines', x:surface.EX, y:surface.EY, z:surface.EZ,
      line:{color:TH().edge, width:1}, name:'Cell edges',
      visible:!orbiting, hoverinfo:'skip', showlegend:false});
  }
  M.wells.forEach(function(w){
    var col = w.sign>0 ? '#3b82f6' : '#ef4444';
    out.push({type:'scatter3d', mode:'lines+markers',
      x:w.points.map(function(p){return p[0];}),
      y:w.points.map(function(p){return p[1];}),
      z:w.points.map(function(p){return p[2];}),
      line:{color:col, width:7}, marker:{size:4, color:col, symbol:'diamond'},
      name:w.name, visible:S.showWells,
      hovertemplate:'<b>'+w.name+'</b><br>'+(w.sign>0?'Injector':'Producer')+'<extra></extra>'});
  });
  if (S.selected!==null){
    out.push({type:'scatter3d', mode:'markers',
      x:[M.centroids[3*S.selected]], y:[M.centroids[3*S.selected+1]],
      z:[M.centroids[3*S.selected+2]],
      marker:{size:9, color:TH().marker, symbol:'diamond-open', line:{width:3}},
      name:'Selected cell', hoverinfo:'skip', showlegend:false});
  }
  return out;
}
function buildLayout(){
  var t = TH();
  var ax = {gridcolor:t.grid, backgroundcolor:t.axisBg, zerolinecolor:t.zero,
            titlefont:{size:11}, tickfont:{size:9.5}};
  return {
    paper_bgcolor:t.paper, plot_bgcolor:t.paper,
    font:{family:'Inter, system-ui, sans-serif', color:t.font, size:11},
    scene:{
      xaxis:Object.assign({title:{text:'Easting'}}, ax),
      yaxis:Object.assign({title:{text:'Northing'}}, ax),
      zaxis:Object.assign({title:{text:'Depth'}, autorange:'reversed'}, ax),
      aspectmode:'manual',
      aspectratio:(function(){
        var a=[BOUNDS[0], BOUNDS[1], BOUNDS[2]*S.ve], m=Math.max(a[0],a[1],a[2]);
        return {x:a[0]/m, y:a[1]/m, z:a[2]/m};
      })(),
      camera:{eye:{x:1.45, y:1.45, z:0.85}}
    },
    legend:{itemsizing:'constant', bgcolor:t.legendBg, font:{size:10.5},
            x:0.99, y:0.01, xanchor:'right', yanchor:'bottom'},
    margin:{l:0, r:0, b:0, t:0},
    uirevision:'pymrst-engineering-view'
  };
}
var CONFIG = {displaylogo:false, responsive:true, scrollZoom:true,
  toImageButtonOptions:{filename:'pymrst-reservoir', scale:2}};

/* ---------------- render ---------------- */
var needSurface = true;
/* gl-line3d re-uploads the whole edge buffer every camera frame, which alone
   took orbiting on a 16k-cell grid from ~120fps to ~36fps. Drop the edges for
   the duration of a drag and put them back on release. */
var edgeTrace = -1, orbiting = false;
/* Plotly emits plotly_click from inside gl3d's own render pass, so a handler
   that re-renders synchronously recurses until the stack blows. render() is
   therefore non-reentrant, and event handlers schedule it instead of calling
   it directly. */
var rendering = false, renderQueued = false;
function render(){
  if (rendering) return;
  rendering = true;
  try {
    if (needSurface){ buildSurface(); needSurface = false; }
    Plotly.react(graph, buildTraces(), buildLayout(), CONFIG);
    updateReadouts();
    drawHistogram();
    drawSeries();
  } finally { rendering = false; }
}
function scheduleRender(){
  if (renderQueued) return;
  renderQueued = true;
  setTimeout(function(){ renderQueued = false; render(); }, 0);
}
function invalidateSurface(){ needSurface = true; }

function updateReadouts(){
  var vals = valuesAt(S.step);
  var shownVals = [];
  for (var c=0;c<NC;c++) if (surface.vis[c] && vals[c]!==null && isFinite(vals[c])) shownVals.push(vals[c]);
  var mm = minMax(shownVals);
  var sorted = shownVals.slice().sort(function(a,b){return a-b;});
  var mean = 0; for (var i=0;i<shownVals.length;i++) mean += shownVals[i];
  mean = shownVals.length ? mean/shownVals.length : NaN;
  $('stat-min').textContent  = shownVals.length ? fmt(mm[0]) : '–';
  $('stat-max').textContent  = shownVals.length ? fmt(mm[1]) : '–';
  $('stat-mean').textContent = shownVals.length ? fmt(mean) : '–';
  $('stat-p90').textContent  = shownVals.length ? fmt(quantile(sorted,0.9)) : '–';
  $('stat-shown').textContent = surface.shown + ' / ' + NC;
  $('stat-wells').textContent = M.wells.length;

  var u = unitLabel();
  $('state-badge').textContent = S.field + (S.delta ? ' Δ' : '') + ' · step ' + S.step
    + (u ? ' · ' + u : '');
  var tu = TIME_UNITS[S.timeUnit];
  $('time-value').textContent = M.times.length
    ? fmt(M.times[Math.min(S.step, M.times.length-1)] * tu[1]) + ' ' + tu[0]
    : 'step ' + S.step + ' / ' + (M.fields[S.field].length-1);

  var r = colorRange();
  if (document.activeElement !== $('cmin')) $('cmin').value = Number(r[0].toPrecision(6));
  if (document.activeElement !== $('cmax')) $('cmax').value = Number(r[1].toPrecision(6));
  $('range-note').textContent = S.rangeMode==='global'
    ? 'Fixed across all ' + M.fields[S.field].length + ' timesteps, so frames stay comparable.'
    : (S.rangeMode==='step' ? 'Rescales every step — playback exaggerates small changes.' : 'Manual limits.');
}

/* ---------------- histogram ---------------- */
function drawHistogram(){
  var cv = $('hist'), ctx = cv.getContext('2d');
  var dpr = window.devicePixelRatio || 1;
  var w = cv.clientWidth, h = 110;
  cv.width = w*dpr; cv.height = h*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  var vals = valuesAt(S.step), data = [];
  for (var c=0;c<NC;c++) if (surface.vis[c] && vals[c]!==null && isFinite(vals[c])) data.push(vals[c]);
  if (!data.length) return;
  var r = colorRange(), lo = r[0], hi = r[1];
  var dm = minMax(data), plo = Math.min(lo, dm[0]), phi = Math.max(hi, dm[1]);
  if (!(phi>plo)) { phi = plo+1; }
  var nb = 44, bins = new Array(nb).fill(0);
  for (var i=0;i<data.length;i++){
    var b = Math.floor((data[i]-plo)/(phi-plo)*nb); if (b<0) b=0; if (b>=nb) b=nb-1; bins[b]++;
  }
  var peak = minMax(bins)[1] || 1;
  var x0 = (lo-plo)/(phi-plo)*w, x1 = (hi-plo)/(phi-plo)*w;
  var t = TH();
  ctx.fillStyle = t.band; ctx.fillRect(x0, 0, Math.max(1,x1-x0), h-14);
  for (var k=0;k<nb;k++){
    var bh = bins[k]/peak*(h-20);
    var bx = k/nb*w, bw = Math.max(1, w/nb - 1);
    var centre = plo + (k+0.5)/nb*(phi-plo);
    ctx.fillStyle = (centre>=lo && centre<=hi) ? t.bar : t.barDim;
    ctx.fillRect(bx, h-14-bh, bw, bh);
  }
  ctx.strokeStyle = t.bar; ctx.lineWidth = 1;
  [x0,x1].forEach(function(x){ ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,h-14); ctx.stroke(); });
  ctx.fillStyle = t.tick; ctx.font = '9px Inter, sans-serif';
  ctx.fillText(fmt(plo), 2, h-3);
  var t = fmt(phi); ctx.fillText(t, w - ctx.measureText(t).width - 2, h-3);
}

/* ---------------- history ---------------- */
function drawSeries(){
  var cv = $('series'), ctx = cv.getContext('2d');
  var dpr = window.devicePixelRatio || 1;
  var w = cv.clientWidth, h = 120;
  cv.width = w*dpr; cv.height = h*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);
  var ns = M.fields[S.field].length;
  var t = TH();
  if (ns < 2){ ctx.fillStyle=t.tick; ctx.font='10px Inter, sans-serif';
    ctx.fillText('single timestep', 6, 18); return; }
  var means = [], cellSeries = [];
  for (var k=0;k<ns;k++){
    var v = valuesAt(k), sum=0, n=0;
    for (var c=0;c<NC;c++){ if (surface.vis[c] && v[c]!==null && isFinite(v[c])){ sum+=v[c]; n++; } }
    means.push(n?sum/n:null);
    if (S.selected!==null) cellSeries.push(v[S.selected]);
  }
  var all = finite(means).concat(finite(cellSeries));
  if (!all.length) return;
  var r = minMax(all), lo=r[0], hi=r[1]; if (!(hi>lo)){ hi=lo+1; }
  var padT=10, padB=16, padL=4, padR=4;
  function px(k){ return padL + k/(ns-1)*(w-padL-padR); }
  function py(v){ return padT + (1-(v-lo)/(hi-lo))*(h-padT-padB); }
  ctx.strokeStyle=t.axisLine; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(padL,h-padB); ctx.lineTo(w-padR,h-padB); ctx.stroke();
  function line(series,color,width){
    ctx.strokeStyle=color; ctx.lineWidth=width; ctx.beginPath();
    var started=false;
    for (var k=0;k<series.length;k++){
      var v=series[k]; if (v===null||!isFinite(v)){ started=false; continue; }
      if (!started){ ctx.moveTo(px(k),py(v)); started=true; } else ctx.lineTo(px(k),py(v));
    }
    ctx.stroke();
  }
  line(means,t.bar,1.8);
  if (S.selected!==null) line(cellSeries,t.marker,1.6);
  var cx = px(S.step);
  ctx.strokeStyle=t.cursor; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(cx,padT-6); ctx.lineTo(cx,h-padB); ctx.stroke();
  ctx.fillStyle=t.tick; ctx.font='9px Inter, sans-serif';
  ctx.fillText(fmt(hi), 3, padT-1);
  ctx.fillText(fmt(lo), 3, h-padB-2);
}

/* ---------------- controls ---------------- */
function fillSelect(el, items, value){
  el.innerHTML = '';
  items.forEach(function(it){
    var o = document.createElement('option');
    o.value = it[0]; o.textContent = it[1];
    el.appendChild(o);
  });
  if (value!==undefined) el.value = value;
}
function syncUnitSelect(){
  var set = UNIT_SETS[M.kinds[S.field]||'generic'];
  fillSelect($('unit'), set.map(function(u,i){ return [String(i), u[0]]; }), String(S.unit[S.field]||0));
  $('unit').disabled = set.length < 2;
}
function setStep(k){
  var ns = M.fields[S.field].length;
  S.step = Math.max(0, Math.min(ns-1, k));
  $('step').value = S.step;
  render();
}

fillSelect($('field'), NAMES.map(function(n){ return [n,n]; }), S.field);
fillSelect($('cmap'), CMAPS.map(function(c){ return [c[0],c[0]]; }), S.cmap);
fillSelect($('time-unit'), TIME_UNITS.map(function(u,i){ return [String(i), u[0]]; }), String(S.timeUnit));
syncUnitSelect();
$('range-mode').value = S.rangeMode;
$('robust').checked = S.robust;
$('reverse').checked = S.reverse;
$('grid-toggle').checked = S.showGrid;
$('well-toggle').checked = S.showWells;
$('opacity').value = Math.round(S.opacity*100);
$('opacity-value').textContent = Math.round(S.opacity*100)+'%';
$('ve').value = S.ve; $('ve-value').textContent = S.ve+'×';
$('step').max = M.fields[S.field].length - 1;
$('time-unit').style.display = M.times.length ? '' : 'none';

$('field').onchange = function(){
  S.field = this.value; S.step = 0; S.manual={lo:null,hi:null};
  S.cmap = M.defaultCmaps[S.field] || S.cmap; $('cmap').value = S.cmap;
  S.reverse = !!(M.defaultReverse && M.defaultReverse[S.field]); $('reverse').checked = S.reverse;
  $('step').max = M.fields[S.field].length - 1; $('step').value = 0;
  syncUnitSelect(); render();
};
$('unit').onchange = function(){ S.unit[S.field] = +this.value; S.manual={lo:null,hi:null}; render(); };
$('cmap').onchange = function(){ S.cmap = this.value; render(); };
function applyTheme(){
  document.documentElement.setAttribute('data-theme', resolvedTheme());
  try { localStorage.setItem('pymrst-theme', S.theme); } catch (e) {}
  render();
}
try {
  var stored = localStorage.getItem('pymrst-theme');
  if (stored === 'auto' || stored === 'light' || stored === 'dark') S.theme = stored;
} catch (e) {}
$('theme').value = S.theme;
$('theme').onchange = function(){ S.theme = this.value; applyTheme(); };
if (window.matchMedia){
  var mq = window.matchMedia('(prefers-color-scheme: light)');
  var onScheme = function(){ if (S.theme === 'auto') applyTheme(); };
  if (mq.addEventListener) mq.addEventListener('change', onScheme);
  else if (mq.addListener) mq.addListener(onScheme);
}
document.documentElement.setAttribute('data-theme', resolvedTheme());
$('reverse').onchange = function(){ S.reverse = this.checked; render(); };
$('delta').onchange = function(){
  S.delta = this.checked; S.manual = {lo:null, hi:null};
  S.cmap = this.checked ? 'RdBu' : (M.defaultCmaps[S.field] || 'Viridis');
  S.reverse = this.checked ? false : !!(M.defaultReverse && M.defaultReverse[S.field]);
  $('cmap').value = S.cmap; $('reverse').checked = S.reverse;
  render();
};
$('range-mode').onchange = function(){ S.rangeMode = this.value; render(); };
$('robust').onchange = function(){ S.robust = this.checked; render(); };
function manualFromInputs(){
  var lo = parseFloat($('cmin').value), hi = parseFloat($('cmax').value);
  if (isFinite(lo) && isFinite(hi) && hi>lo){
    S.manual = {lo:lo, hi:hi}; S.rangeMode='manual'; $('range-mode').value='manual'; render();
  }
}
$('cmin').onchange = manualFromInputs;
$('cmax').onchange = manualFromInputs;

$('opacity').oninput = function(){
  S.opacity = +this.value/100; $('opacity-value').textContent = this.value+'%'; render();
};
$('ve').oninput = function(){
  S.ve = +this.value; $('ve-value').textContent = this.value+'×'; render();
};
$('grid-toggle').onchange = function(){ S.showGrid = this.checked; render(); };
$('well-toggle').onchange = function(){ S.showWells = this.checked; render(); };
$('time-unit').onchange = function(){ S.timeUnit = +this.value; updateReadouts(); };

/* index filters */
if (M.cartDims && M.ijk){
  var dims = M.cartDims;
  S.filt = {i:[0,dims[0]-1], j:[0,dims[1]-1], k:[0,dims[2]-1]};
  ['i','j','k'].forEach(function(axis, ai){
    var a = $(axis+'0'), b = $(axis+'1');
    a.min = b.min = 0; a.max = b.max = dims[ai]-1;
    a.value = 0; b.value = dims[ai]-1;
    function apply(){
      var lo = Math.min(+a.value, +b.value), hi = Math.max(+a.value, +b.value);
      S.filt[axis] = [lo,hi];
      $(axis+'-label').textContent = lo+'–'+hi;
      invalidateSurface(); render();
    }
    a.oninput = apply; b.oninput = apply;
    $(axis+'-label').textContent = '0–'+(dims[ai]-1);
  });
  $('filter-reset').onclick = function(){
    ['i','j','k'].forEach(function(axis, ai){
      $(axis+'0').value = 0; $(axis+'1').value = dims[ai]-1;
      S.filt[axis] = [0, dims[ai]-1];
      $(axis+'-label').textContent = '0–'+(dims[ai]-1);
    });
    invalidateSurface(); render();
  };
  $('filter-top').onclick = function(){
    $('k0').value = 0; $('k1').value = 0; S.filt.k = [0,0];
    $('k-label').textContent = '0–0';
    invalidateSurface(); render();
  };
} else {
  $('filter-section').innerHTML =
    '<div class="section-title">Index filter</div>'
    + '<div class="help">This grid has no logical (I, J, K) indexing, so index filtering is unavailable.</div>';
}

/* transport */
$('step').oninput = function(){ setStep(+this.value); };
$('first').onclick = function(){ setStep(0); };
$('last').onclick  = function(){ setStep(M.fields[S.field].length-1); };
$('previous').onclick = function(){ setStep(S.step-1); };
$('next').onclick = function(){ setStep(S.step+1); };
function togglePlay(){
  if (S.timer){ clearInterval(S.timer); S.timer=null; S.playing=false;
    $('play').innerHTML='&#9654;'; return; }
  var ns = M.fields[S.field].length;
  if (ns<2) return;
  S.playing = true; $('play').innerHTML='&#10073;&#10073;';
  S.timer = setInterval(function(){ setStep((S.step+1)%ns); }, 620);
}
$('play').onclick = togglePlay;

/* cameras */
var CAMERAS = {iso:{x:1.45,y:1.45,z:0.85}, top:{x:0,y:0,z:2.6},
               front:{x:0,y:2.6,z:0.1}, side:{x:2.6,y:0,z:0.1}};
Array.prototype.forEach.call(document.querySelectorAll('[data-camera]'), function(b){
  b.onclick = function(){ Plotly.relayout(graph, {'scene.camera.eye': CAMERAS[b.dataset.camera]}); };
});

/* panels */
function panelToggle(btnId, panelId){
  $(btnId).onclick = function(){
    var open = $(panelId).classList.toggle('open');
    this.setAttribute('aria-expanded', String(open));
  };
}
panelToggle('rail-toggle', 'rail');
panelToggle('inspector-toggle', 'inspector');

/* selection — Plotly only attaches its emitter once the div has been plotted,
   so this is bound after the first render(), not before. */
function bindPlotEvents(){
graph.addEventListener('pointerdown', function(){
  if (edgeTrace < 0 || orbiting) return;
  orbiting = true;
  Plotly.restyle(graph, {visible:false}, [edgeTrace]);
});
window.addEventListener('pointerup', function(){
  if (!orbiting) return;
  orbiting = false;
  if (edgeTrace >= 0) Plotly.restyle(graph, {visible:true}, [edgeTrace]);
});
graph.on('plotly_click', function(e){
  var p = e.points && e.points[0];
  if (!p || !p.customdata) return;
  var c = Math.round(p.customdata[0]);
  S.selected = c;
  var vals = valuesAt(S.step);
  var ijk = M.ijk
    ? 'I ' + M.ijk[3*c] + ' &middot; J ' + M.ijk[3*c+1] + ' &middot; K ' + M.ijk[3*c+2] + '<br>'
    : '';
  $('selection').innerHTML = '<b>Cell ' + c + '</b><br>' + ijk
    + 'X ' + fmt(M.centroids[3*c]) + ' &middot; Y ' + fmt(M.centroids[3*c+1]) + '<br>'
    + 'Depth ' + fmt(M.centroids[3*c+2]) + '<br>'
    + axisLabel() + ' <b>' + fmt(vals[c]) + '</b>';
  $('series-cell').textContent = 'cell ' + c;
  scheduleRender();
});
}

/* keyboard */
document.addEventListener('keydown', function(e){
  var tag = (e.target.tagName||'').toLowerCase();
  if (tag==='input' || tag==='select' || tag==='textarea') return;
  if (e.key==='ArrowRight'){ setStep(S.step+1); e.preventDefault(); }
  else if (e.key==='ArrowLeft'){ setStep(S.step-1); e.preventDefault(); }
  else if (e.key===' '){ togglePlay(); e.preventDefault(); }
  else if (e.key==='r' || e.key==='R'){
    Plotly.relayout(graph, {'scene.camera.eye': CAMERAS.iso});
  }
});

window.addEventListener('resize', function(){ drawHistogram(); drawSeries(); });

render();
bindPlotEvents();
window.__PYMRST_VIEWER__ = {state:S, render:render, setStep:setStep,
  invalidateSurface:invalidateSurface, colorRange:colorRange, surface:function(){return surface;}};
})();
"""


__all__ = [
    "BrowserReservoirPayload",
    "plot_reservoir_browser",
    "plot_saved_states_browser",
    "reservoir_browser_data",
]
