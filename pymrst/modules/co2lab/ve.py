"""Backward-compatible import path for vertical-equilibrium CO2Lab helpers."""

from pymrst.modules.co2lab_ve.ve import (
    TopSurfaceGrid,
    VEFluid,
    VEModel,
    VEState,
    average_rock,
    top_surface_grid,
)

__all__ = [
    "TopSurfaceGrid",
    "VEFluid",
    "VEModel",
    "VEState",
    "average_rock",
    "top_surface_grid",
]
