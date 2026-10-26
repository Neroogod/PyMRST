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
    griddim: int = 3
    cell_pos: np.ndarray | None = None
    cell_nodes: np.ndarray | None = None


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
    well_series: Mapping[str, np.ndarray] | None = None,
    rock=None,
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
    pore_volume = None
    if rock is not None:
        payload.fields.update(_rock_fields(rock, g.cells.num))
        pore_volume = _pore_volume(g, rock)
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
        well_series=well_series,
        pore_volume=pore_volume,
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
    "ratio": "Portland",
    "generic": "Viridis",
}


#: Display names for the well series a run usually carries.
_SERIES_LABEL = {
    "bhp": "Bottom-hole pressure",
    "qws": "Water rate",
    "qos": "Oil rate",
    "qgs": "Gas rate",
    "watercut": "Water cut",
    "gor": "Gas-oil ratio",
    "cumoil": "Cumulative oil",
    "cumwater": "Cumulative water",
    "cumgas": "Cumulative gas",
    "fieldoil": "Field cumulative oil",
    "fieldwater": "Field cumulative water",
    "fieldgas": "Field cumulative gas",
}


def _series_kind(name: str) -> str:
    """Classify a well time series so the viewer can offer sensible units."""
    n = str(name).strip().lower()
    if n.startswith("cum") or n.startswith("field"):
        return "volume_gas" if "gas" in n else "volume_liquid"
    if "bhp" in n or "pres" in n:
        return "pressure"
    if "cut" in n:
        return "fraction"
    if "gor" in n:
        return "gor"
    if n.startswith("qg") or "gas" in n:
        return "rate_gas"
    if n.startswith("q") or "rate" in n:
        return "rate_liquid"
    return "generic"


def _cumulative(rates: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Trapezoidal time-integral of a rate series.

    Magnitudes are integrated, so a producer accumulates volume produced and an
    injector volume injected; the sign convention lives in the well role, not in
    the cumulative.
    """
    r = np.abs(np.asarray(rates, dtype=float))
    t = np.asarray(times, dtype=float).ravel()
    out = np.zeros_like(r)
    if r.shape[0] < 2 or t.size != r.shape[0]:
        return out
    dt = np.diff(t)[:, None]
    out[1:] = np.cumsum(0.5 * (r[1:] + r[:-1]) * dt, axis=0)
    return out


def _derived_well_series(
    series: dict[str, np.ndarray], times: np.ndarray | None
) -> dict[str, tuple[np.ndarray, str]]:
    """Water cut, GOR and cumulative volumes, where the inputs allow it.

    Returns ``name -> (values, scope)`` where scope is ``"well"`` or ``"field"``.
    """
    lower = {k.lower(): v for k, v in series.items()}
    qw, qo, qg = lower.get("qws"), lower.get("qos"), lower.get("qgs")
    out: dict[str, tuple[np.ndarray, str]] = {}
    with np.errstate(divide="ignore", invalid="ignore"):
        if qw is not None and qo is not None:
            aw, ao = np.abs(qw), np.abs(qo)
            total = aw + ao
            out["watercut"] = (
                np.where(total > 1e-30, aw / np.where(total > 0, total, 1.0), np.nan), "well")
        if qg is not None and qo is not None:
            ao = np.abs(qo)
            out["gor"] = (
                np.where(ao > 1e-30, np.abs(qg) / np.where(ao > 0, ao, 1.0), np.nan), "well")

    if times is None or np.asarray(times).size < 2:
        return out
    for key, name in (("qos", "cumOil"), ("qws", "cumWater"), ("qgs", "cumGas")):
        if key not in lower:
            continue
        cum = _cumulative(lower[key], times)
        out[name] = (cum, "well")
        # Field total: one line, summed across wells.
        out["field" + name[3:]] = (cum.sum(axis=1)[:, None], "field")
    return out


def _well_series_payload(
    well_series, wells: list[dict], n_steps: int, times: np.ndarray | None = None
) -> dict:
    """Normalise well time series to (n_steps, n_wells) and tag them."""
    if not well_series:
        return {}
    n_wells = len(wells)
    clean: dict[str, np.ndarray] = {}
    for name, values in dict(well_series).items():
        arr = np.asarray(values, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.shape[0] != n_steps or arr.shape[1] != n_wells:
            raise ValueError(
                f"well series {name!r} has shape {arr.shape}, expected "
                f"({n_steps}, {n_wells})"
            )
        clean[str(name)] = arr

    scoped: dict[str, tuple[np.ndarray, str]] = {k: (v, "well") for k, v in clean.items()}
    scoped.update(_derived_well_series(clean, times))
    return {
        name: {
            "label": _SERIES_LABEL.get(name.lower(), name),
            "kind": _series_kind(name),
            "scope": scope,
            "values": _json_array(arr),
        }
        for name, (arr, scope) in scoped.items()
    }


def _rock_fields(rock, n_cells: int) -> dict[str, np.ndarray]:
    """Expose static rock properties as single-step fields."""
    out: dict[str, np.ndarray] = {}
    poro = np.asarray(getattr(rock, "poro", None), dtype=float).ravel()
    if poro.size == n_cells:
        out["porosity"] = poro[None, :]
    perm = np.asarray(getattr(rock, "perm", None), dtype=float)
    if perm.ndim == 1 and perm.size == n_cells:
        out["permX"] = perm[None, :]
    elif perm.ndim == 2 and perm.shape[0] == n_cells:
        for i, name in enumerate(("permX", "permY", "permZ")[: perm.shape[1]]):
            out[name] = perm[:, i][None, :]
    return out


def _pore_volume(g: Grid, rock) -> list | None:
    """Per-cell pore volume, so the viewer can report volumes under a filter."""
    poro = np.asarray(getattr(rock, "poro", None), dtype=float).ravel()
    if poro.size != g.cells.num:
        return None
    ntg = getattr(rock, "ntg", None)
    ntg = 1.0 if ntg is None else np.asarray(ntg, dtype=float).ravel()
    pv = np.asarray(g.cells.volumes, dtype=float).ravel() * poro * ntg
    return [0.0 if not np.isfinite(v) else float(v) for v in pv]


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
    if n in {"rs", "rv"} or "gor" in n:
        return "ratio"
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
    well_series: Mapping[str, np.ndarray] | None = None,
    pore_volume: list | None = None,
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
        "griddim": int(payload.griddim),
        "cellPos": None if payload.cell_pos is None else [int(v) for v in payload.cell_pos],
        "cellNodes": None if payload.cell_nodes is None else [int(v) for v in payload.cell_nodes],
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
        "wellSeries": _well_series_payload(
            well_series,
            payload.wells,
            0 if payload.times is None else int(payload.times.size),
            payload.times,
        ),
        "poreVolume": pore_volume,
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
  <div class="viewer-error" id="viewer-error" hidden></div>
  <button class="panel-toggle rail-toggle" id="rail-toggle" aria-label="Toggle controls" aria-expanded="false">&#9776;</button>
  <button class="panel-toggle inspector-toggle" id="inspector-toggle" aria-label="Toggle statistics" aria-expanded="false">&#9878;</button>
  <div class="vgrip vgrip-left" id="rail-grip" title="Drag to resize · double-click to reset"></div>
  <div class="vgrip vgrip-right" id="inspector-grip" title="Drag to resize · double-click to reset"></div>
  <div class="topbar"><div class="title-card"><strong>{safe_title}</strong>
    <span class="badge" id="state-badge"></span>
    <div class="help">Drag to orbit &middot; scroll to zoom &middot; click a cell to inspect &middot; &larr;/&rarr; step &middot; space play</div></div>
  </div>
  <div id="pymrst-reservoir-view"></div>
  <div class="curves" id="curves" hidden>
    <div class="curves-grip" id="curves-grip" title="Drag to resize · double-click to reset"></div>
    <div class="curves-head"><strong>Well curves</strong>
      <select id="curve-series"></select><select id="curve-unit"></select>
      <span class="axis-ctl">
        <select id="curve-axis" title="Choose which axis to adjust">
          <option value="y">Y axis</option><option value="x">X axis</option></select>
        <input id="curve-amin" type="number" step="any" placeholder="auto">
        <span class="dash">–</span>
        <input id="curve-amax" type="number" step="any" placeholder="auto">
        <button id="curve-aauto" title="Auto-fit this axis">Auto</button>
        <button id="curve-csv" title="Download this series as CSV">CSV</button>
        <label class="check" id="curve-log-wrap"><input id="curve-log" type="checkbox"> Log</label></span>
      <button id="curves-hide" class="curves-hide" title="Hide curves">&#9662;</button></div>
    <canvas id="curve-canvas"></canvas>
  </div>
  <div class="transport">
    <button id="first" title="First step">&#9198;</button>
    <button id="previous" title="Previous step">&#9664;</button>
    <button id="play" class="primary" title="Play/pause">&#9654;</button>
    <button id="next" title="Next step">&#9654;</button>
    <button id="last" title="Last step">&#9197;</button>
    <button id="curves-toggle" title="Show well curves">Curves</button>
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
      <div class="stat"><b id="stat-pv">&ndash;</b><span>Pore volume [m&sup3;]</span></div>
      <div class="stat"><b id="stat-hcpv">&ndash;</b><span>HC pore vol [m&sup3;]</span></div>
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
    griddim = int(getattr(g, "griddim", 3) or 3)
    cell_pos, cell_nodes = _cell_polygons(g) if griddim == 2 else (None, None)
    return {
        "nodes": coords,
        "face_pos": np.asarray(g.faces.nodePos, dtype=np.int64),
        "face_nodes": np.asarray(g.faces.nodes, dtype=np.int64),
        "face_neighbors": np.asarray(g.faces.neighbors, dtype=np.int64).reshape(-1, 2),
        "ijk": _cell_ijk(g),
        "cart_dims": tuple(int(v) for v in g.cartDims) if getattr(g, "cartDims", None) is not None else None,
        "griddim": griddim,
        "cell_pos": cell_pos,
        "cell_nodes": cell_nodes,
    }


def _cell_polygons(g: Grid) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell node rings for 2-D grids, angularly sorted around the centroid.

    A 2-D grid's faces are edges - two nodes each - not polygons, so the
    face-boundary algorithm the browser uses to skin a 3-D solid has nothing
    to triangulate: every face gets treated as degenerate and dropped. A flat
    grid has no interior to hide in the first place, so every cell is its own
    visible polygon; the angular sort works for any convex cell shape
    (Cartesian, PEBI, ...) without assuming a particular face ordering.
    """
    cell_face_pos = np.asarray(g.cells.facePos, dtype=np.int64)
    cell_faces = np.asarray(g.cells.faces, dtype=np.int64)
    face_node_pos = np.asarray(g.faces.nodePos, dtype=np.int64)
    face_nodes = np.asarray(g.faces.nodes, dtype=np.int64)
    coords = np.asarray(g.nodes.coords, dtype=float)
    centroids = np.asarray(g.cells.centroids, dtype=float)
    cell_pos = [0]
    cell_nodes: list[int] = []
    for c in range(g.cells.num):
        faces = cell_faces[cell_face_pos[c]:cell_face_pos[c + 1]]
        nodes: set[int] = set()
        for f in faces:
            nodes.update(int(n) for n in face_nodes[face_node_pos[f]:face_node_pos[f + 1]])
        if len(nodes) < 3:
            cell_pos.append(cell_pos[-1])
            continue
        cx, cy = float(centroids[c, 0]), float(centroids[c, 1])
        ordered = sorted(nodes, key=lambda n: np.arctan2(coords[n, 1] - cy, coords[n, 0] - cx))
        cell_nodes.extend(ordered)
        cell_pos.append(len(cell_nodes))
    return np.asarray(cell_pos, dtype=np.int64), np.asarray(cell_nodes, dtype=np.int64)


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


#: Per-well time series a saved run typically stores, keyed by array name.
_WELL_SERIES_KEYS = ("bhp", "qWs", "qOs", "qGs")


def saved_well_series(
    saved: Mapping[str, np.ndarray],
    well_names: Sequence[str] | None = None,
) -> dict[str, np.ndarray]:
    """Pull the per-well (n_steps, n_wells) time series out of a saved run.

    Series columns follow the order the run was saved in. When ``well_names``
    is given, columns are permuted to match it, so curves cannot be attributed
    to the wrong well if the wells were rebuilt in a different order.
    """
    out: dict[str, np.ndarray] = {}
    for key in _WELL_SERIES_KEYS:
        if key not in saved:
            continue
        arr = np.asarray(saved[key], dtype=float)
        if arr.ndim == 2:
            out[key] = arr
    if not out or well_names is None or "well_names" not in saved:
        return out

    saved_names = [str(n) for n in np.asarray(saved["well_names"]).ravel()]
    wanted = [str(n) for n in well_names]
    if saved_names == wanted:
        return out
    try:
        order = [saved_names.index(n) for n in wanted]
    except ValueError:
        # A name we cannot match: leave the columns alone rather than guess.
        return out
    return {key: arr[:, order] for key, arr in out.items()}


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
        well_series=saved_well_series(
            saved, [str(getattr(w, "name", f"W{i + 1}")) for i, w in enumerate(wells or [])]
        ),
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
.shell{display:grid;grid-template-columns:var(--rail-w,296px) 1fr var(--insp-w,288px);height:100vh}
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
.viewer-error{position:absolute;left:0;right:0;top:0;z-index:9;padding:9px 14px;
background:#7f1d1d;color:#fee2e2;font-size:12px;border-bottom:1px solid #b91c1c}
.viewer-error[hidden]{display:none}
.topbar{position:absolute;left:16px;top:12px;z-index:3;pointer-events:none}
.title-card{background:var(--overlay);border:1px solid var(--line);border-radius:10px;padding:9px 12px;
backdrop-filter:blur(10px);box-shadow:var(--shadow);max-width:min(460px,52vw)}
.badge{display:inline-block;padding:2px 8px;border-radius:99px;background:var(--accent-soft);
color:var(--accent);font-size:10px;margin-left:6px;font-variant-numeric:tabular-nums}
.transport{display:flex;align-items:center;gap:8px;padding:9px 14px;border-top:1px solid var(--line);
background:var(--panel)}
.transport button{padding:5px 9px;min-width:34px}
.transport button.primary{background:var(--accent);border-color:var(--accent);color:#fff}
.transport button.active{background:var(--accent-soft);border-color:var(--accent);color:var(--accent)}
.transport #step{flex:1}
.transport select{width:auto;padding:5px 7px}
.time-value{color:var(--accent);font-variant-numeric:tabular-nums;min-width:118px;text-align:right;font-size:12px}
.curves{border-top:1px solid var(--line);background:var(--panel);padding:10px 14px 12px;
position:relative;display:flex;flex-direction:column;height:230px;min-height:150px}
/* A class rule beats the UA [hidden] rule, so display:none must be restated. */
.curves[hidden]{display:none}
.vgrip{position:absolute;top:0;bottom:0;width:9px;cursor:ew-resize;z-index:7;touch-action:none}
.vgrip-left{left:-4px}.vgrip-right{right:-4px}
.vgrip::after{content:'';position:absolute;top:50%;left:3px;transform:translateY(-50%);
width:3px;height:52px;border-radius:2px;background:var(--line);transition:background .15s}
.vgrip:hover::after,.vgrip.dragging::after{background:var(--accent)}
.curves-grip{position:absolute;top:-4px;left:0;right:0;height:9px;cursor:ns-resize;z-index:6;
touch-action:none}
.curves-grip::after{content:'';position:absolute;left:50%;top:3px;transform:translateX(-50%);
width:52px;height:3px;border-radius:2px;background:var(--line);transition:background .15s}
.curves-grip:hover::after,.curves-grip.dragging::after{background:var(--accent)}
.curves-head{display:flex;align-items:center;gap:10px;margin-bottom:8px;flex:none;flex-wrap:wrap}
.axis-ctl{display:inline-flex;align-items:center;gap:6px}
.axis-ctl label{margin:0;font-size:11px}
.axis-ctl input[type=number]{width:82px;padding:5px 6px}
.axis-ctl .dash{color:var(--muted)}
.axis-ctl button{padding:5px 9px}
.axis-ctl .check{gap:5px;white-space:nowrap}
.curves-head strong{font-size:10px;text-transform:uppercase;letter-spacing:.12em;color:var(--accent)}
.curves-head select{width:auto;padding:5px 7px}
.curves-head button{padding:4px 9px}
.curves-hide{margin-left:auto}
#curve-canvas{flex:1;min-height:0;height:auto;cursor:crosshair;background:transparent}
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
@media(max-width:1180px){.shell{grid-template-columns:var(--rail-w,296px) 1fr}
#inspector-grip{display:none}.inspector{position:absolute;right:0;top:0;
width:288px;height:100%;transform:translateX(100%);transition:.2s;border-left:1px solid var(--line);z-index:5}
.inspector.open{transform:none}
.inspector-toggle{display:grid;place-items:center;position:absolute;top:12px;right:12px;z-index:6;
width:36px;height:36px;padding:0;font-size:16px;background:var(--overlay);border:1px solid var(--line);
border-radius:9px;backdrop-filter:blur(10px)}
.topbar{right:58px}}
@media(max-width:820px){.shell{grid-template-columns:1fr}
#rail-grip{display:none}
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
var $ = function(id){ return document.getElementById(id); };
window.addEventListener('error', function(e){
  var b = $('viewer-error');
  if (!b) return;
  b.textContent = 'Viewer error: ' + (e.message || 'script failure')
    + ' — some controls will not respond. See the browser console for details.';
  b.hidden = false;
});
var M = JSON.parse(document.getElementById('pymrst-model').textContent);
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
  ratio:       [['sm3/sm3',1,0],['scf/stb',5.6145833,0]],
  generic:     [['--',1,0]]
};
var DEFAULT_UNIT = {pressure:1, saturation:0, permeability:1, porosity:0,
                    temperature:1, ratio:1, generic:0};
var TIME_UNITS = [['s',1],['hours',1/3600],['days',1/86400],['years',1/(86400*365.25)]];
/* Well time series carry their own unit families: rates are volumetric, and
   gas is conventionally reported per Mscf rather than per m3. */
var SERIES_UNITS = {
  pressure:    [['Pa',1,0],['bar',1e-5,0],['MPa',1e-6,0],['psi',1.4503773773e-4,0]],
  rate_liquid: [['m3/s',1,0],['m3/day',86400,0],['stb/day',86400*6.2898107,0]],
  rate_gas:    [['m3/s',1,0],['m3/day',86400,0],['Mscf/day',86400*35.3146667/1000,0]],
  fraction:    [['fraction',1,0],['%',100,0]],
  gor:         [['m3/m3',1,0],['scf/stb',5.6145833,0]],
  volume_liquid:[['m3',1,0],['stb',6.2898107,0],['Mstb',6.2898107e-3,0]],
  volume_gas:  [['m3',1,0],['Mscf',35.3146667e-3,0],['MMscf',35.3146667e-6,0]],
  generic:     [['--',1,0]]
};
var SERIES_DEFAULT_UNIT = {pressure:1, rate_liquid:1, rate_gas:1, fraction:1,
                           gor:1, volume_liquid:1, volume_gas:1, generic:0};
var INJ_SHADES  = ['#60a5fa','#2563eb','#93c5fd','#1e40af'];
var PROD_SHADES = ['#f87171','#dc2626','#fca5a5','#991b1b'];
var CURVE_NAMES = Object.keys(M.wellSeries || {});
/* Static properties (porosity, permeability) carry a single step. They must not
   shorten the timeline, so the slider spans the longest field. */
var NSTEPS = 1;
NAMES.forEach(function(n){ NSTEPS = Math.max(NSTEPS, M.fields[n].length); });
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
  return {lo:lo, hi:hi,
          span:[Math.max(hi[0]-lo[0],1e-9), Math.max(hi[1]-lo[1],1e-9), Math.max(hi[2]-lo[2],1e-9)]};
})();
/* Wells are drawn rising above the reservoir top so they stay readable against
   opaque rock; the headroom is a fraction of gross thickness. */
var WELL_HEAD_FRAC = 0.5;
var WELL_HEAD_Z = BOUNDS.lo[2] - WELL_HEAD_FRAC * BOUNDS.span[2];

/* Plotly and the 2D canvases paint their own pixels, so the theme cannot be
   expressed in CSS alone -- these mirror the stylesheet variables. */
var THEMES = {
  dark:  {paper:'#060b13', axisBg:'#0a1424', grid:'#20304a', zero:'#2b3f5c',
          font:'#dbeafe', edge:'rgba(148,163,184,0.18)', legendBg:'rgba(8,17,31,.72)',
          canvas:'#101d31', axisLine:'#1e2d44', bar:'#38bdf8', barDim:'#31445f',
          band:'rgba(56,189,248,0.10)', tick:'#7e91ab', marker:'#fbbf24', cursor:'rgba(226,236,249,0.45)',
          inj:'#60a5fa', prod:'#f87171',
          plotBg:'#0a1424', gridLine:'rgba(126,145,171,0.22)', frame:'#22344e',
          legendFill:'rgba(10,20,36,0.82)', text:'#dbeafe'},
  light: {paper:'#eef2f7', axisBg:'#ffffff', grid:'#d7e0ea', zero:'#b9c6d6',
          font:'#17324d', edge:'rgba(71,85,105,0.30)', legendBg:'rgba(255,255,255,.85)',
          canvas:'#f4f7fb', axisLine:'#d7e0ea', bar:'#0284c7', barDim:'#c3d0de',
          band:'rgba(2,132,199,0.12)', tick:'#5c6c81', marker:'#b45309', cursor:'rgba(15,29,46,0.35)',
          inj:'#1d4ed8', prod:'#b91c1c',
          plotBg:'#ffffff', gridLine:'rgba(92,108,129,0.20)', frame:'#d7e0ea',
          legendFill:'rgba(255,255,255,0.86)', text:'#17324d'}
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
  filt: null, selected: null, playing: false, timer: null, camera: null,
  curve: CURVE_NAMES[0] || null, curveUnit: 0, curvesOpen: false,
  curveY: {min:null, max:null, log:false}, curveX: {min:null, max:null},
  curveAxis: 'y'
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
function finishSurface(vis, shown, X, Y, Z, I, J, K, owner, EX, EY, EZ){
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
function buildSurface(){
  var vis = computeVisible();
  var shown=0; for (var c=0;c<NC;c++) if (vis[c]) shown++;
  if (M.griddim === 2 && M.cellPos) return buildSurface2D(vis, shown);
  var N=M.nodes, FP=M.facePos, FN=M.faceNodes, NB=M.faceNeighbors;
  var X=[],Y=[],Z=[],I=[],J=[],K=[],owner=[],EX=[],EY=[],EZ=[];
  var NN=N.length/3, seenEdge=new Set();
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
  finishSurface(vis, shown, X, Y, Z, I, J, K, owner, EX, EY, EZ);
}
/* A 2-D grid's faces are edges, not polygons: there is no boundary-face skin
   to compute, because there is no interior to hide in a flat sheet. Every
   visible cell draws its own pre-ordered polygon (see _cell_polygons in the
   Python writer) instead of the face/neighbour test buildSurface() uses. */
function buildSurface2D(vis, shown){
  var N=M.nodes, CP=M.cellPos, CN=M.cellNodes;
  var X=[],Y=[],Z=[],I=[],J=[],K=[],owner=[],EX=[],EY=[],EZ=[];
  var NN=N.length/3, seenEdge=new Set();
  for (var c=0;c<NC;c++){
    if (!vis[c]) continue;
    var s=CP[c], e=CP[c+1], m=e-s;
    if (m<3) continue;
    var base=X.length;
    for (var p=s;p<e;p++){
      var n=CN[p];
      X.push(N[3*n]); Y.push(N[3*n+1]); Z.push(N[3*n+2]); owner.push(c);
    }
    for (var t=1;t<m-1;t++){ I.push(base); J.push(base+t); K.push(base+t+1); }
    for (var q=0;q<m;q++){
      var na=CN[s+q], nb=CN[s+((q+1)%m)];
      var key = na<nb ? na*NN+nb : nb*NN+na;
      if (seenEdge.has(key)) continue;
      seenEdge.add(key);
      EX.push(N[3*na],N[3*nb],null); EY.push(N[3*na+1],N[3*nb+1],null);
      EZ.push(N[3*na+2],N[3*nb+2],null);
    }
  }
  finishSurface(vis, shown, X, Y, Z, I, J, K, owner, EX, EY, EZ);
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
      hoverinfo:'skip', showlegend:false});
  }
  if (S.showWells) out = out.concat(wellTraces());
  if (S.selected!==null){
    out.push({type:'scatter3d', mode:'markers',
      x:[M.centroids[3*S.selected]], y:[M.centroids[3*S.selected+1]],
      z:[M.centroids[3*S.selected+2]],
      marker:{size:9, color:TH().marker, symbol:'diamond-open', line:{width:3}},
      name:'Selected cell', hoverinfo:'skip', showlegend:false});
  }
  return out;
}
function wellTraces(){
  var t = TH(), out = [];
  M.wells.forEach(function(w){
    var inj = w.sign > 0;
    var col = inj ? t.inj : t.prod;
    var tag = inj ? 'INJ' : 'PROD';
    var role = inj ? 'Injector' : 'Producer';
    var nm = String(w.name);
    /* Do not stutter when the name already carries the role. */
    var label = nm.toUpperCase().indexOf(tag) === 0 ? nm : tag + ' · ' + nm;
    var xs = w.points.map(function(p){return p[0];});
    var ys = w.points.map(function(p){return p[1];});
    var zs = w.points.map(function(p){return p[2];});
    var hov = '<b>' + nm + '</b><br>' + role + '<br>'
      + w.points.length + ' completion' + (w.points.length === 1 ? '' : 's')
      + '<extra></extra>';

    out.push({type:'scatter3d', mode:'lines',
      x:[xs[0]].concat(xs), y:[ys[0]].concat(ys), z:[WELL_HEAD_Z].concat(zs),
      line:{color:col, width:6}, name:label, legendgroup:nm,
      hovertemplate:hov});
    out.push({type:'scatter3d', mode:'markers', x:xs, y:ys, z:zs,
      marker:{size:6, color:col, symbol:'diamond'},
      legendgroup:nm, showlegend:false, hovertemplate:hov});
    out.push({type:'scatter3d', mode:'markers+text',
      x:[xs[0]], y:[ys[0]], z:[WELL_HEAD_Z],
      marker:{size:7, color:col, symbol: inj ? 'diamond' : 'circle', line:{width:1}},
      text:[label], textposition:'top center',
      textfont:{size:11, color:col, family:'Inter, system-ui, sans-serif'},
      legendgroup:nm, showlegend:false, hoverinfo:'skip'});
  });
  return out;
}
var DEFAULT_CAMERA = {eye:{x:1.45, y:1.45, z:0.85}};
/* Plotly.react re-applies whatever camera the layout carries, so the layout has
   to carry the camera the user is currently looking through -- not a constant. */
function currentCamera(){
  try {
    var c = graph._fullLayout && graph._fullLayout.scene && graph._fullLayout.scene.camera;
    if (c && c.eye) return {eye:{x:c.eye.x, y:c.eye.y, z:c.eye.z},
                            up:c.up, center:c.center, projection:c.projection};
  } catch (e) {}
  return S.camera || DEFAULT_CAMERA;
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
        var zs = BOUNDS.span[2] * (M.wells.length ? 1 + WELL_HEAD_FRAC : 1);
        var a=[BOUNDS.span[0], BOUNDS.span[1], zs*S.ve], m=Math.max(a[0],a[1],a[2]);
        return {x:a[0]/m, y:a[1]/m, z:a[2]/m};
      })(),
      camera: currentCamera()
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
/* De-duplicating shared edges cut gl-line3d's per-frame cost from ~20ms to
   ~2ms, so edges now stay visible while orbiting. Hiding them on pointerdown/up
   re-plotted the scene before gl3d had committed the dragged camera, which
   snapped the view back to its previous orientation. */
var edgeTrace = -1;
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
    drawCurves();
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
  /* Volumes are reported over the cells actually shown, so an index filter
     reports the region's volume rather than the whole model's. */
  if (M.poreVolume){
    var pv = 0, hc = 0;
    var swName = null;
    for (var f2=0;f2<NAMES.length;f2++){
      if (M.kinds[NAMES[f2]] === 'saturation' && /w/i.test(NAMES[f2])){ swName = NAMES[f2]; break; }
    }
    var sw0 = swName ? M.fields[swName][0] : null;
    for (var c3=0;c3<NC;c3++){
      if (!surface.vis[c3]) continue;
      var v3 = M.poreVolume[c3];
      if (!isFinite(v3)) continue;
      pv += v3;
      if (sw0){ var s3 = sw0[c3]; if (s3 !== null && isFinite(s3)) hc += v3*(1-s3); }
    }
    $('stat-pv').textContent = fmt(pv);
    $('stat-hcpv').textContent = sw0 ? fmt(hc) : '–';
  } else {
    $('stat-pv').textContent = '–'; $('stat-hcpv').textContent = '–';
  }

  var u = unitLabel();
  $('state-badge').textContent = S.field + (S.delta ? ' Δ' : '') + ' · step ' + S.step
    + (u ? ' · ' + u : '');
  var tu = TIME_UNITS[S.timeUnit];
  $('time-value').textContent = M.times.length
    ? fmt(M.times[Math.min(S.step, M.times.length-1)] * tu[1]) + ' ' + tu[0]
    : 'step ' + S.step + ' / ' + (NSTEPS-1);

  var r = colorRange();
  if (document.activeElement !== $('cmin')) $('cmin').value = Number(r[0].toPrecision(6));
  if (document.activeElement !== $('cmax')) $('cmax').value = Number(r[1].toPrecision(6));
  $('range-note').textContent = S.rangeMode==='global'
    ? 'Fixed across all ' + M.fields[S.field].length + ' timestep(s), so frames stay comparable.'
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

/* ---------------- well curves ---------------- */
var CURVE_COLORS = (function(){
  var seen = {inj:0, prod:0};
  return M.wells.map(function(w){
    var inj = w.sign > 0, key = inj ? 'inj' : 'prod';
    var shades = inj ? INJ_SHADES : PROD_SHADES;
    return shades[(seen[key]++) % shades.length];
  });
})();
function curveUnitSet(){
  var spec = M.wellSeries[S.curve];
  return SERIES_UNITS[(spec && spec.kind) || 'generic'] || SERIES_UNITS.generic;
}
function curveUnit(){ return curveUnitSet()[S.curveUnit] || curveUnitSet()[0]; }
var CURVE_PAD = {l:56, r:14, t:10, b:24};
var curveXDomain = null;
var syncCurveAxisInputs = null;

function drawCurves(){
  if (!S.curvesOpen || !S.curve) return;
  var cv = $('curve-canvas'), ctx = cv.getContext('2d'), t = TH();
  var dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
  if (!w || !h) return;
  cv.width = w*dpr; cv.height = h*dpr; ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);

  var spec = M.wellSeries[S.curve], u = curveUnit();
  var vals = spec.values, ns = vals.length, nw = M.wells.length;
  if (!ns) return;
  var log = S.curveY.log;
  /* In log mode non-positive samples are gaps, matching how plotting libraries
     treat them, rather than silently clamping them to a floor. */
  var conv = function(v){
    if (v === null || !isFinite(v)) return null;
    var d = v*u[1] + u[2];
    if (!log) return d;
    return d > 0 ? Math.log10(d) : null;
  };
  var fwd = function(v){
    if (v === null || !isFinite(v)) return null;
    return log ? (v > 0 ? Math.log10(v) : null) : v;
  };
  var inv = function(v){ return log ? Math.pow(10, v) : v; };

  var cols = spec.scope === 'field' ? 1 : nw;
  var lo = Infinity, hi = -Infinity;
  for (var k=0;k<ns;k++){ for (var i=0;i<cols;i++){ var v = conv(vals[k][i]);
    if (v === null) continue; if (v<lo) lo=v; if (v>hi) hi=v; } }
  if (!isFinite(lo)){ lo = 0; hi = 1; }
  if (!(hi>lo)) hi = lo + Math.max(Math.abs(lo)*0.1, 1);
  var pad = (hi-lo)*0.08; lo -= pad; hi += pad;

  var umin = fwd(S.curveY.min), umax = fwd(S.curveY.max);
  if (umin !== null) lo = umin;
  if (umax !== null) hi = umax;
  if (!(hi > lo)) hi = lo + 1;

  /* Timesteps are usually non-uniform, so x must be positioned by time value.
     Positioning by step index silently distorts the curves. */
  var tu = TIME_UNITS[S.timeUnit];
  var hasT = M.times.length > 0;
  var tAt = function(k){ return hasT ? M.times[Math.min(k, M.times.length-1)]*tu[1] : k; };
  var t0 = tAt(0), t1 = tAt(ns-1);
  if (S.curveX.min !== null) t0 = S.curveX.min;
  if (S.curveX.max !== null) t1 = S.curveX.max;
  if (!(t1 > t0)) t1 = t0 + 1;
  curveXDomain = {t0:t0, t1:t1, hasT:hasT, scale:tu[1]};

  var P = CURVE_PAD;
  var plotW = w - P.l - P.r, plotH = h - P.t - P.b;
  var px = function(k){ return P.l + (tAt(k) - t0)/(t1 - t0)*plotW; };
  var py = function(v){ return P.t + (1-(v-lo)/(hi-lo))*plotH; };

  /* plotting surface */
  ctx.fillStyle = t.plotBg;
  ctx.fillRect(P.l, P.t, plotW, plotH);

  ctx.font = '9.5px Inter, system-ui, sans-serif';
  ctx.lineWidth = 1;

  /* horizontal grid + y labels */
  var YT = 5;
  ctx.textAlign = 'right'; ctx.textBaseline = 'middle';
  for (var q=0;q<=YT;q++){
    var vv = lo + q/YT*(hi-lo), yy = Math.round(py(vv)) + 0.5;
    ctx.strokeStyle = t.gridLine;
    ctx.beginPath(); ctx.moveTo(P.l, yy); ctx.lineTo(P.l+plotW, yy); ctx.stroke();
    ctx.fillStyle = t.tick;
    ctx.fillText(fmt(inv(vv)), P.l-7, yy);
  }

  /* vertical grid + time labels, evenly spaced in time */
  var XT = 6;
  ctx.textBaseline = 'top';
  for (var c=0;c<=XT;c++){
    var frac = c/XT, tv = t0 + frac*(t1-t0);
    var xx = Math.round(P.l + frac*plotW) + 0.5;
    ctx.strokeStyle = t.gridLine;
    ctx.beginPath(); ctx.moveTo(xx, P.t); ctx.lineTo(xx, P.t+plotH); ctx.stroke();
    if (c === 0 || c === XT || c === XT/2){
      ctx.textAlign = c === 0 ? 'left' : (c === XT ? 'right' : 'center');
      ctx.fillStyle = t.tick;
      var base = hasT ? fmt(tv) : ('step ' + Math.round(tv));
      ctx.fillText(c === XT ? base + ' ' + (hasT ? tu[0] : '') : base, xx, P.t+plotH+6);
    }
  }

  /* zero line reads differently from the grid */
  if (!log && lo < 0 && hi > 0){
    ctx.strokeStyle = t.tick; ctx.globalAlpha = 0.7;
    var zy = Math.round(py(0)) + 0.5;
    ctx.beginPath(); ctx.moveTo(P.l, zy); ctx.lineTo(P.l+plotW, zy); ctx.stroke();
    ctx.globalAlpha = 1;
  }

  /* series, clipped so a manual range cannot draw outside the axes */
  ctx.save();
  ctx.beginPath(); ctx.rect(P.l, P.t, plotW, plotH); ctx.clip();
  var isField = spec.scope === 'field';
  var lineCount = isField ? 1 : nw;
  for (var i2=0;i2<lineCount;i2++){
    ctx.strokeStyle = isField ? t.bar : CURVE_COLORS[i2];
    ctx.lineWidth = 1.8; ctx.beginPath();
    var started = false;
    for (var k2=0;k2<ns;k2++){
      var v2 = conv(vals[k2][i2]);
      if (v2 === null){ started = false; continue; }
      if (!started){ ctx.moveTo(px(k2), py(v2)); started = true; }
      else ctx.lineTo(px(k2), py(v2));
    }
    ctx.stroke();
  }
  var cx = Math.round(px(Math.min(S.step, ns-1))) + 0.5;
  ctx.strokeStyle = t.cursor; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(cx, P.t); ctx.lineTo(cx, P.t+plotH); ctx.stroke();
  ctx.restore();

  /* frame above the grid */
  ctx.strokeStyle = t.frame; ctx.lineWidth = 1;
  ctx.strokeRect(P.l+0.5, P.t+0.5, plotW-1, plotH-1);

  /* Legend inside the plot, in whichever top corner the data sits lowest, so
     it does not collide with the curves. */
  ctx.font = '10px Inter, system-ui, sans-serif';
  var names = spec.scope === 'field'
    ? ['Field total'] : M.wells.map(function(wl){ return wl.name; });
  var nameW = 0;
  for (var n2=0;n2<names.length;n2++) nameW = Math.max(nameW, ctx.measureText(names[n2]).width);
  var lineH = 15, boxW = nameW + 40, boxH = names.length*lineH + 10;
  if (boxW < plotW - 20 && boxH < plotH - 20){
    /* Score all four corners by how far the nearest curve is, in that half of
       the plot, from that corner -- then take the roomiest. */
    var cut = Math.max(1, Math.floor(ns*0.35));
    var lMax = -Infinity, lMin = Infinity, rMax = -Infinity, rMin = Infinity;
    for (var i3=0;i3<cols;i3++){
      for (var k3=0;k3<cut;k3++){ var lv = conv(vals[k3][i3]);
        if (lv === null) continue;
        if (lv > lMax) lMax = lv; if (lv < lMin) lMin = lv; }
      for (var k4=ns-cut;k4<ns;k4++){ var rv = conv(vals[k4][i3]);
        if (rv === null) continue;
        if (rv > rMax) rMax = rv; if (rv < rMin) rMin = rv; }
    }
    var span = hi - lo;
    var corners = [
      {x:'l', y:'t', gap: isFinite(lMax) ? hi - lMax : span},
      {x:'r', y:'t', gap: isFinite(rMax) ? hi - rMax : span},
      {x:'l', y:'b', gap: isFinite(lMin) ? lMin - lo : span},
      {x:'r', y:'b', gap: isFinite(rMin) ? rMin - lo : span}
    ];
    var best = corners[0];
    for (var c2=1;c2<corners.length;c2++) if (corners[c2].gap > best.gap) best = corners[c2];
    var bx = best.x === 'l' ? P.l + 10 : P.l + plotW - boxW - 10;
    var by = best.y === 't' ? P.t + 10 : P.t + plotH - boxH - 10;
    ctx.fillStyle = t.legendFill;
    ctx.fillRect(bx, by, boxW, boxH);
    ctx.strokeStyle = t.frame; ctx.lineWidth = 1;
    ctx.strokeRect(bx+0.5, by+0.5, boxW-1, boxH-1);
    ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    for (var i4=0;i4<names.length;i4++){
      var ly = by + 5 + lineH*i4 + lineH/2;
      ctx.strokeStyle = spec.scope === 'field' ? t.bar : CURVE_COLORS[i4];
      ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(bx+8, ly); ctx.lineTo(bx+26, ly); ctx.stroke();
      ctx.fillStyle = t.text || t.tick;
      ctx.fillText(names[i4], bx+31, ly);
    }
  }
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
  S.step = Math.max(0, Math.min(NSTEPS-1, k));
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
$('step').max = NSTEPS - 1;
$('time-unit').style.display = M.times.length ? '' : 'none';

$('field').onchange = function(){
  S.field = this.value; S.manual={lo:null,hi:null};
  S.cmap = M.defaultCmaps[S.field] || S.cmap; $('cmap').value = S.cmap;
  S.reverse = !!(M.defaultReverse && M.defaultReverse[S.field]); $('reverse').checked = S.reverse;
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
$('time-unit').onchange = function(){
  S.timeUnit = +this.value;
  if (S.curveX){ S.curveX.min = null; S.curveX.max = null; }
  if (typeof syncCurveAxisInputs === 'function') syncCurveAxisInputs();
  updateReadouts(); drawCurves();
};

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

/* well curves wiring */
function syncCurveUnits(){
  var set = curveUnitSet();
  fillSelect($('curve-unit'), set.map(function(u,i){ return [String(i), u[0]]; }), String(S.curveUnit));
  $('curve-unit').disabled = set.length < 2;
}
function setCurvesOpen(open){
  S.curvesOpen = open;
  $('curves').hidden = !open;
  $('curves-toggle').classList.toggle('active', open);
  $('curves-toggle').setAttribute('aria-expanded', String(open));
  if (open) drawCurves();
  Plotly.Plots.resize(graph);
}
if (!CURVE_NAMES.length){
  $('curves-toggle').hidden = true;
} else {
  fillSelect($('curve-series'),
    CURVE_NAMES.map(function(n){ return [n, M.wellSeries[n].label || n]; }), S.curve);
  S.curveUnit = SERIES_DEFAULT_UNIT[M.wellSeries[S.curve].kind || 'generic'] || 0;
  syncCurveUnits();
  /* One pair of inputs edits whichever axis the selector names; each axis keeps
     its own limits so switching back shows what you set. */
  function axisState(which){ return (which || S.curveAxis) === 'x' ? S.curveX : S.curveY; }
  function syncAxisInputs(){
    var isX = S.curveAxis === 'x', st = axisState();
    $('curve-amin').value = st.min === null ? '' : st.min;
    $('curve-amax').value = st.max === null ? '' : st.max;
    var unit = isX
      ? (M.times.length ? TIME_UNITS[S.timeUnit][0] : 'step')
      : curveUnit()[0];
    var ph = 'auto' + (unit && unit !== '--' ? ' (' + unit + ')' : '');
    $('curve-amin').placeholder = ph;
    $('curve-amax').placeholder = ph;
    $('curve-log-wrap').style.display = isX ? 'none' : '';
  }
  function resetCurveAxis(which){
    var st = axisState(which);
    st.min = null; st.max = null;
    syncAxisInputs();
  }
  function readAxis(){
    var st = axisState();
    var lo = parseFloat($('curve-amin').value), hi = parseFloat($('curve-amax').value);
    st.min = isFinite(lo) ? lo : null;
    st.max = isFinite(hi) ? hi : null;
    drawCurves();
  }
  $('curve-csv').onclick = function(){
    var spec = M.wellSeries[S.curve], u = curveUnit();
    var isField = spec.scope === 'field';
    var cols = isField ? ['Field total'] : M.wells.map(function(wl){ return wl.name; });
    var tu = TIME_UNITS[S.timeUnit];
    var rows = ['time [' + (M.times.length ? tu[0] : 'step') + '],'
      + cols.map(function(c){ return '"' + c + ' [' + u[0] + ']"'; }).join(',')];
    for (var k=0;k<spec.values.length;k++){
      var tv = M.times.length ? M.times[Math.min(k, M.times.length-1)]*tu[1] : k;
      var cells = [tv];
      for (var i=0;i<cols.length;i++){
        var raw = spec.values[k][i];
        cells.push(raw === null || !isFinite(raw) ? '' : (raw*u[1] + u[2]));
      }
      rows.push(cells.join(','));
    }
    var blob = new Blob([rows.join('\n')], {type:'text/csv;charset=utf-8'});
    var a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'pymrst-' + S.curve + '.csv';
    document.body.appendChild(a); a.click();
    setTimeout(function(){ URL.revokeObjectURL(a.href); a.remove(); }, 0);
  };
  syncCurveAxisInputs = syncAxisInputs;
  $('curve-axis').onchange = function(){ S.curveAxis = this.value; syncAxisInputs(); };
  $('curve-amin').onchange = readAxis;
  $('curve-amax').onchange = readAxis;
  $('curve-aauto').onclick = function(){ resetCurveAxis(); drawCurves(); };
  $('curve-series').onchange = function(){
    S.curve = this.value;
    S.curveUnit = SERIES_DEFAULT_UNIT[M.wellSeries[S.curve].kind || 'generic'] || 0;
    resetCurveAxis('y'); syncCurveUnits(); syncAxisInputs(); drawCurves();
  };
  /* A typed range is in display units, so changing units invalidates it. */
  $('curve-unit').onchange = function(){
    S.curveUnit = +this.value; resetCurveAxis('y'); syncAxisInputs(); drawCurves();
  };
  $('curve-log').onchange = function(){
    S.curveY.log = this.checked; resetCurveAxis('y'); drawCurves();
  };
  syncAxisInputs();
  $('curves-toggle').onclick = function(){ setCurvesOpen(!S.curvesOpen); };

  makeDragResize($('curves-grip'), {
    axis:'y', sign:-1, min:150, def:230, key:'pymrst-curves-h',
    max: function(){
      return Math.max(150, document.querySelector('.viewer').getBoundingClientRect().height - 190);
    },
    get: function(){ return $('curves').getBoundingClientRect().height; },
    set: function(v){ $('curves').style.height = v + 'px'; },
    redraw: drawCurves
  });
  $('curves-hide').onclick = function(){ setCurvesOpen(false); };
  /* Clicking the chart scrubs time, so the 3D state follows the curve. */
  $('curve-canvas').onclick = function(e){
    var r = this.getBoundingClientRect();
    var ns = M.wellSeries[S.curve].values.length;
    if (ns < 2 || !curveXDomain) return;
    var f = (e.clientX - r.left - CURVE_PAD.l) / (r.width - CURVE_PAD.l - CURVE_PAD.r);
    f = Math.max(0, Math.min(1, f));
    /* x is a time position, so pick the step nearest that time. */
    var target = curveXDomain.t0 + f*(curveXDomain.t1 - curveXDomain.t0);
    var best = 0, bestD = Infinity;
    for (var k=0;k<ns;k++){
      var tk = curveXDomain.hasT
        ? M.times[Math.min(k, M.times.length-1)]*curveXDomain.scale : k;
      var d = Math.abs(tk - target);
      if (d < bestD){ bestD = d; best = k; }
    }
    setStep(best);
  };
}

/* panel resizing --------------------------------------------------------- */
function redrawCanvases(){ drawHistogram(); drawSeries(); drawCurves(); }

/* One drag implementation for both axes: cfg.sign maps pointer travel to the
   direction the panel should grow. */
function makeDragResize(grip, cfg){
  if (!grip) return;
  var startPos = 0, startVal = 0, dragging = false;
  function apply(v, persist){
    var clamped = Math.max(cfg.min, Math.min(cfg.max(), v));
    cfg.set(clamped);
    if (cfg.redraw) cfg.redraw();
    if (persist){
      try { localStorage.setItem(cfg.key, String(Math.round(clamped))); } catch (e) {}
    }
  }
  try {
    var saved = parseFloat(localStorage.getItem(cfg.key));
    if (isFinite(saved)) apply(saved, false);
  } catch (e) {}
  grip.addEventListener('pointerdown', function(e){
    dragging = true;
    startPos = cfg.axis === 'x' ? e.clientX : e.clientY;
    startVal = cfg.get();
    grip.classList.add('dragging');
    try { grip.setPointerCapture(e.pointerId); } catch (_) {}
    e.preventDefault();
  });
  grip.addEventListener('pointermove', function(e){
    if (!dragging) return;
    var now = cfg.axis === 'x' ? e.clientX : e.clientY;
    apply(startVal + (now - startPos) * cfg.sign, false);
  });
  function end(e){
    if (!dragging) return;
    dragging = false;
    grip.classList.remove('dragging');
    try { grip.releasePointerCapture(e.pointerId); } catch (_) {}
    apply(cfg.get(), true);
  }
  grip.addEventListener('pointerup', end);
  grip.addEventListener('pointercancel', end);
  grip.addEventListener('dblclick', function(){ apply(cfg.def, true); });
}

/* Always leave the 3D view at least this wide when sizing the side panels. */
var VIEWER_MIN_W = 360;
function railWidth(){ return $('rail').getBoundingClientRect().width; }
function inspectorWidth(){ return $('inspector').getBoundingClientRect().width; }

makeDragResize($('rail-grip'), {
  axis:'x', sign:1, min:220, def:296, key:'pymrst-rail-w',
  max: function(){ return Math.max(220, window.innerWidth - inspectorWidth() - VIEWER_MIN_W); },
  get: railWidth,
  set: function(v){ document.documentElement.style.setProperty('--rail-w', v + 'px'); },
  redraw: redrawCanvases
});
makeDragResize($('inspector-grip'), {
  axis:'x', sign:-1, min:220, def:288, key:'pymrst-insp-w',
  max: function(){ return Math.max(220, window.innerWidth - railWidth() - VIEWER_MIN_W); },
  get: inspectorWidth,
  set: function(v){ document.documentElement.style.setProperty('--insp-w', v + 'px'); },
  redraw: redrawCanvases
});

/* transport */
$('step').oninput = function(){ setStep(+this.value); };
$('first').onclick = function(){ setStep(0); };
$('last').onclick  = function(){ setStep(NSTEPS-1); };
$('previous').onclick = function(){ setStep(S.step-1); };
$('next').onclick = function(){ setStep(S.step+1); };
function togglePlay(){
  if (S.timer){ clearInterval(S.timer); S.timer=null; S.playing=false;
    $('play').innerHTML='&#9654;'; return; }
  if (NSTEPS<2) return;
  S.playing = true; $('play').innerHTML='&#10073;&#10073;';
  S.timer = setInterval(function(){ setStep((S.step+1)%NSTEPS); }, 620);
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
graph.on('plotly_relayout', function(e){
  if (e['scene.camera']) S.camera = e['scene.camera'];
  else if (e['scene.camera.eye']) S.camera = {eye: e['scene.camera.eye']};
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

window.addEventListener('resize', function(){ drawHistogram(); drawSeries(); drawCurves(); });

render();
bindPlotEvents();
if (window.ResizeObserver){
  var roPending = false;
  new ResizeObserver(function(){
    if (roPending) return;
    roPending = true;
    requestAnimationFrame(function(){ roPending = false; Plotly.Plots.resize(graph); });
  }).observe(graph);
}
window.__PYMRST_VIEWER__ = {state:S, render:render, setStep:setStep,
  invalidateSurface:invalidateSurface, colorRange:colorRange, surface:function(){return surface;}};
})();
"""


__all__ = [
    "BrowserReservoirPayload",
    "plot_reservoir_browser",
    "plot_saved_states_browser",
    "reservoir_browser_data",
    "saved_well_series",
]
