"""Validate browser plotting data preparation and optional HTML rendering."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.params import add_well, make_rock
from pymrst.plotting.browser import plot_reservoir_browser, reservoir_browser_data


def main() -> int:
    g = compute_geometry(cart_grid([3, 2, 1], [30.0, 20.0, 5.0]))
    rock = make_rock(g, 1e-13, 0.2)
    wells = add_well(
        None,
        g,
        rock,
        np.array([0, 3]),
        well_type="rate",
        val=1.0,
        compi=np.array([1.0, 0.0]),
        sign=1,
        name="Injector",
    )
    fields = {
        "pressure": np.vstack([
            np.linspace(100.0, 200.0, g.cells.num),
            np.linspace(110.0, 210.0, g.cells.num),
        ]),
        "sW": np.full((2, g.cells.num), 0.2),
    }
    payload = reservoir_browser_data(g, fields, wells=wells, times=[0.0, 1.0], zscale=4.0)
    ok = (
        payload.centroids.shape == (g.cells.num, 3)
        and payload.grid_lines.shape[1] == 3
        and payload.mesh_points.shape[1] == 3
        and payload.mesh_i.size > 0
        and payload.mesh_cell_index.size == payload.mesh_points.shape[0]
        and payload.fields["pressure"].shape == (2, g.cells.num)
        and len(payload.wells) == 1
    )
    print(f"{'ok  ' if ok else 'FAIL'} browser payload")
    failed = 0 if ok else 1

    try:
        import plotly  # noqa: F401
    except ImportError:
        print("skip browser HTML render: plotly is not installed")
        return failed

    out = Path(tempfile.gettempdir()) / "pymrst_browser_plot_validation.html"
    path = plot_reservoir_browser(
        g,
        fields,
        wells=wells,
        times=[0.0, 1.0],
        filename=out,
        zscale=4.0,
    )
    html_ok = path.exists() and path.stat().st_size > 1000
    print(f"{'ok  ' if html_ok else 'FAIL'} browser HTML render: {path}")
    return failed + (0 if html_ok else 1)


if __name__ == "__main__":
    raise SystemExit(main())
