"""MRST ``plotGridDeformed.m`` deformed-grid handle."""

import numpy as np

from pymrst.modules.vemmech.plotting import VEMPlotHandle, deformed_grid


def plot_grid_deformed_geometry(g, u):
    """Return the deformed grid geometry."""
    return deformed_grid(g, u)


def plot_grid_deformed(g, u, backend: str = "geometry", **backend_kwargs):
    """Return a handle for MRST ``plotGridDeformed``."""
    return VEMPlotHandle(
        "grid_deformed",
        plot_grid_deformed_geometry,
        g,
        u=np.asarray(u, dtype=float),
        backend=backend,
        backend_kwargs=backend_kwargs,
    )

__all__ = ["plot_grid_deformed", "plot_grid_deformed_geometry"]
