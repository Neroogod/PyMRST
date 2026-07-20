"""Backward-compatible import path for the AD-core aquifer model.

The implementation lives in `ad_core.models.aquifers.aquifer`, mirroring
MRST's `ad-core/models/aquifers/AquiferModel.m`.
"""

from pymrst.modules.ad_core.models.aquifers.aquifer import (
    AquiferConnections,
    AquiferModel,
    compute_aquifer_fluxes,
    equilibrium_aquifer_pressure,
    fetkovich_coefficient,
    update_aquifer_state,
)

__all__ = [
    "AquiferConnections",
    "AquiferModel",
    "compute_aquifer_fluxes",
    "equilibrium_aquifer_pressure",
    "fetkovich_coefficient",
    "update_aquifer_state",
]
