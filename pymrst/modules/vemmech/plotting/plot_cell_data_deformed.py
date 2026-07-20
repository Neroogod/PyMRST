"""MRST ``plotCellDataDeformed.m`` deformed cell-data handle."""

import numpy as np

from pymrst.modules.vemmech.plotting import VEMPlotHandle, deformed_grid


def plot_cell_data_deformed_geometry(g, data, u):
    """Return deformed grid geometry with cell data."""
    return {"grid": deformed_grid(g, u), "data": np.asarray(data)}


def plot_cell_data_deformed(
    g, data, u, backend: str = "geometry", **backend_kwargs
):
    """Return a handle for MRST ``plotCellDataDeformed``."""
    return VEMPlotHandle(
        "cell_data_deformed",
        plot_cell_data_deformed_geometry,
        g,
        u=np.asarray(u, dtype=float),
        data=np.asarray(data),
        backend=backend,
        backend_kwargs=backend_kwargs,
    )

__all__ = ["plot_cell_data_deformed", "plot_cell_data_deformed_geometry"]
