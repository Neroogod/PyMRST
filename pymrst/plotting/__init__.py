"""Visualization of grids and cell/face data (MRST plotting/), on PyVista."""

import os
from pathlib import Path
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pymrst_matplotlib"))

from pymrst.plotting.plot_grid import (
    plot_cell_data,
    plot_grid,
    plot_well,
    to_pyvista,
)
from pymrst.plotting.plot_toolbar import (
    PlotToolbar,
    apply_transform,
    normalize_data,
    plot_toolbar,
    states_to_fields,
)
from pymrst.plotting.plot_inventory import plot_trapping_inventory
from pymrst.plotting.plot_streamlines import (
    plot_streamlines,
    streamlines_to_pyvista,
)
from pymrst.plotting.plot_ve import plot_caprock, plot_plume, plot_traps

__all__ = [
    "to_pyvista", "plot_grid", "plot_cell_data", "plot_well",
    "plot_caprock", "plot_plume", "plot_traps",
    "plot_streamlines", "streamlines_to_pyvista",
    "plot_toolbar", "PlotToolbar", "normalize_data", "states_to_fields",
    "apply_transform",
    "plot_trapping_inventory",
    "load_saved_state_fields", "saved_cell_fields", "plot_saved_state",
    "plot_toolbar_saved_states",
    "plot_reservoir_browser", "plot_saved_states_browser",
    "reservoir_browser_data", "BrowserReservoirPayload",
]


def __getattr__(name):
    if name in {
        "load_saved_state_fields",
        "saved_cell_fields",
        "plot_saved_state",
        "plot_toolbar_saved_states",
    }:
        from pymrst.plotting import plot_saved_states as saved

        return getattr(saved, name)
    if name in {
        "plot_reservoir_browser",
        "plot_saved_states_browser",
        "reservoir_browser_data",
        "BrowserReservoirPayload",
    }:
        from pymrst.plotting import browser

        return getattr(browser, name)
    raise AttributeError(name)
