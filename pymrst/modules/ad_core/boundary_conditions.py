"""Backward-compatible import path for the AD-core boundary conditions.

The implementation lives in `ad_core.utils.boundary_conditions`, mirroring
MRST's `ad-core/utils/addFluxesFromSourcesAndBC.m`.
"""

from pymrst.modules.ad_core.utils.boundary_conditions import (
    add_bc_and_sources,
    boundary_reservoir_fluxes,
    source_reservoir_fluxes,
)

__all__ = [
    "add_bc_and_sources",
    "boundary_reservoir_fluxes",
    "source_reservoir_fluxes",
]
