"""Plot saved simulation states without rerunning the simulator."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyvista as pv

from pymrst.gridprocessing import compute_geometry, tensor_grid
from pymrst.gridprocessing.grid_structure import Cells, Faces, Grid, Nodes
from pymrst.modules.deckformat import read_spe1_deck
from pymrst.plotting.plot_grid import plot_cell_data, plot_well
from pymrst.plotting.plot_toolbar import plot_toolbar
from pymrst.params import NORM_GRAVITY, add_well, make_rock
from pymrst.params.wells_and_bc import Well
from pymrst.utils.units import DAY

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SPE1_DECK = ROOT / "mrst-2024a-Geomech/examples/data/SPE1/BENCH_SPE1.DATA"


def build_spe1_plot_context(deck_path: str | Path | None = None):
    """Rebuild the SPE1 grid, rock, and wells needed for saved-state plots."""
    deck = read_spe1_deck(Path(deck_path) if deck_path is not None else DEFAULT_SPE1_DECK)
    g = compute_geometry(tensor_grid(deck["x"], deck["y"], deck["z"]))
    rock = make_rock(g, deck["perm"], deck["poro"][:, None])
    wells = None
    for name in ("PRODUCER", "INJECTOR"):
        wd = deck["wells"][name]
        wells = add_well(
            wells,
            g,
            rock,
            wd["cells"],
            well_type=wd["type"],
            val=wd["val"],
            radius=wd["radius"],
            compi=wd["compi"],
            sign=wd["sign"],
            name=name,
            ref_depth=wd["ref_depth"],
            gravity=np.array([0.0, 0.0, NORM_GRAVITY]),
        )
    return g, rock, wells


def load_saved_state_fields(path: str | Path) -> dict[str, np.ndarray]:
    """Load arrays from a ``save_tutorial_states`` .npz file."""
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key].copy() for key in data.files}


def grid_from_saved_metadata(saved: dict[str, np.ndarray]) -> Grid:
    """Reconstruct a PyMRST grid from ``save_tutorial_states`` metadata."""
    return Grid(
        cells=Cells(
            num=int(saved["grid.cells.num"]),
            facePos=np.asarray(saved["grid.cells.facePos"], dtype=np.int64),
            faces=np.asarray(saved["grid.cells.faces"], dtype=np.int64),
            indexMap=np.asarray(saved["grid.cells.indexMap"], dtype=np.int64),
            volumes=np.asarray(saved["grid.cells.volumes"], dtype=float),
            centroids=np.asarray(saved["grid.cells.centroids"], dtype=float),
        ),
        faces=Faces(
            num=int(saved["grid.faces.num"]),
            nodePos=np.asarray(saved["grid.faces.nodePos"], dtype=np.int64),
            nodes=np.asarray(saved["grid.faces.nodes"], dtype=np.int64),
            neighbors=np.asarray(saved["grid.faces.neighbors"], dtype=np.int64),
            areas=np.asarray(saved["grid.faces.areas"], dtype=float),
            normals=np.asarray(saved["grid.faces.normals"], dtype=float),
            centroids=np.asarray(saved["grid.faces.centroids"], dtype=float),
        ),
        nodes=Nodes(
            num=int(saved["grid.nodes.num"]),
            coords=np.asarray(saved["grid.nodes.coords"], dtype=float),
        ),
        griddim=int(np.asarray(saved["grid.griddim"]).item()),
        cartDims=_cart_dims_from_saved(saved),
    )


def wells_from_saved_metadata(saved: dict[str, np.ndarray]) -> list[Well]:
    """Reconstruct enough well metadata for plotting from saved states."""
    wells = []
    i = 0
    while f"well.{i}.cells" in saved:
        prefix = f"well.{i}"
        wells.append(
            Well(
                cells=np.asarray(saved[f"{prefix}.cells"], dtype=np.int64),
                type=str(np.asarray(saved.get(f"{prefix}.type", "unknown")).item()),
                val=float(np.asarray(saved.get(f"{prefix}.val", 0.0)).item()),
                r=np.asarray(saved.get(f"{prefix}.r", np.zeros(1)), dtype=float),
                dir=str(np.asarray(saved.get(f"{prefix}.dir", "z")).item()),
                WI=np.asarray(saved.get(f"{prefix}.WI", np.zeros(1)), dtype=float),
                dZ=np.asarray(saved.get(f"{prefix}.dZ", np.zeros(1)), dtype=float),
                name=str(np.asarray(saved.get(f"{prefix}.name", f"W{i + 1}")).item()),
                compi=np.asarray(saved.get(f"{prefix}.compi", np.ones(1)), dtype=float),
                refDepth=float(np.asarray(saved.get(f"{prefix}.refDepth", 0.0)).item()),
                sign=int(np.asarray(saved.get(f"{prefix}.sign", 0)).item()),
            )
        )
        i += 1
    return wells


def _cart_dims_from_saved(saved: dict[str, np.ndarray]) -> tuple[int, ...] | None:
    dims = np.asarray(saved.get("grid.cartDims", [])).ravel()
    return tuple(int(v) for v in dims) or None


def saved_cell_fields(saved: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return only time-varying cell fields suitable for plotToolbar."""
    nsteps = int(np.asarray(saved["time"]).size) if "time" in saved else None
    fields = {}
    for key, value in saved.items():
        arr = np.asarray(value)
        if arr.ndim == 2 and arr.shape[1] > 2 and (nsteps is None or arr.shape[0] == nsteps):
            fields[key] = arr
    if "pressure" in fields:
        fields["pressure_bar"] = fields["pressure"] / 1e5
    if "sW" in fields and "sG" in fields:
        fields["sO"] = 1.0 - fields["sW"] - fields["sG"]
    return fields


def plot_saved_state(
    state_file: str | Path,
    *,
    field: str = "sG",
    step: int = -1,
    deck_path: str | Path | None = None,
    filename: str | Path | None = None,
    show: bool = True,
    zscale: float = 1.0,
    show_edges: bool = False,
    cmap: str = "viridis",
):
    """Plot one saved cell field at one report step."""
    saved = load_saved_state_fields(state_file)
    if field not in saved:
        raise KeyError(f"{field!r} not found in {state_file}")
    data = np.asarray(saved[field], dtype=float)
    if data.ndim != 2:
        raise ValueError(f"{field!r} is not a time-by-cell field")
    if "grid.cells.num" in saved:
        g = grid_from_saved_metadata(saved)
        wells = wells_from_saved_metadata(saved)
    else:
        g, _, wells = build_spe1_plot_context(deck_path)
    values = data[int(step)]
    pl = pv.Plotter(off_screen=filename is not None)
    plot_cell_data(
        g,
        values,
        label=field,
        cmap=cmap,
        show_edges=show_edges,
        plotter=pl,
        zscale=zscale,
    )
    plot_well(g, wells, pl, zscale=zscale)
    if filename is not None:
        pl.screenshot(str(filename))
        pl.close()
        return Path(filename)
    if show:
        pl.show()
    return pl


def plot_toolbar_saved_states(
    state_file: str | Path,
    *,
    deck_path: str | Path | None = None,
    show: bool = True,
    zscale: float = 1.0,
):
    """Open plotToolbar for saved SPE1 state histories."""
    saved = load_saved_state_fields(state_file)
    g = grid_from_saved_metadata(saved) if "grid.cells.num" in saved else build_spe1_plot_context(deck_path)[0]
    if not np.isclose(float(zscale), 1.0):
        g = _scaled_grid_for_toolbar(g, zscale)
    return plot_toolbar(
        g,
        saved_cell_fields(saved),
        show=show,
        times=saved.get("time"),
        time_unit=DAY,
        time_label="days",
        title=f"Saved states: {Path(state_file).name}",
    )


def _scaled_grid_for_toolbar(g, zscale: float):
    import copy

    gg = copy.deepcopy(g)
    coords = gg.nodes.coords.astype(float).copy()
    if coords.shape[1] == 2:
        coords = np.column_stack([coords, np.zeros(coords.shape[0])])
    z0 = float(np.min(coords[:, 2]))
    coords[:, 2] = z0 + (coords[:, 2] - z0) * float(zscale)
    gg.nodes.coords = coords

    centroids = gg.cells.centroids.astype(float).copy()
    if centroids.shape[1] == 2:
        centroids = np.column_stack([centroids, np.zeros(centroids.shape[0])])
    z0 = float(np.min(centroids[:, 2]))
    centroids[:, 2] = z0 + (centroids[:, 2] - z0) * float(zscale)
    gg.cells.centroids = centroids
    return gg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot saved PyMRST state histories.")
    parser.add_argument("state_file", type=Path)
    parser.add_argument("--field", default="sG", help="Saved cell field to plot.")
    parser.add_argument("--step", type=int, default=-1, help="Report-step index.")
    parser.add_argument("--deck", type=Path, default=None, help="Optional SPE1 deck path.")
    parser.add_argument("--output", type=Path, default=None, help="Optional PNG filename.")
    parser.add_argument("--toolbar", action="store_true", help="Open plotToolbar.")
    parser.add_argument("--browser", action="store_true", help="Write a browser-based interactive HTML viewer.")
    parser.add_argument("--browser-style", choices=("mesh", "points"), default="mesh", help="Browser rendering style.")
    parser.add_argument("--browser-all-fields", action="store_true", help="Include every saved cell field in browser frames.")
    parser.add_argument("--open-browser", action="store_true", help="Open the generated browser viewer.")
    parser.add_argument("--list-fields", action="store_true", help="List saved cell fields and exit.")
    parser.add_argument("--zscale", type=float, default=1.0, help="Vertical exaggeration.")
    parser.add_argument("--show-edges", action="store_true")
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args(argv)

    if args.list_fields:
        fields = saved_cell_fields(load_saved_state_fields(args.state_file))
        for name, values in fields.items():
            print(f"{name}: {values.shape}")
        return 0

    if args.toolbar:
        plot_toolbar_saved_states(
            args.state_file,
            deck_path=args.deck,
            show=not args.no_show,
            zscale=args.zscale,
        )
    elif args.browser:
        from pymrst.plotting.browser import plot_saved_states_browser

        out = args.output or args.state_file.with_suffix(".browser.html")
        path = plot_saved_states_browser(
            args.state_file,
            filename=out,
            deck_path=args.deck,
            zscale=args.zscale,
            style=args.browser_style,
            field=args.field,
            all_fields=args.browser_all_fields,
            open_browser=args.open_browser,
        )
        print(f"Wrote {path}")
    else:
        out = plot_saved_state(
            args.state_file,
            field=args.field,
            step=args.step,
            deck_path=args.deck,
            filename=args.output,
            show=not args.no_show,
            zscale=args.zscale,
            show_edges=args.show_edges,
        )
        if args.output is not None:
            print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
