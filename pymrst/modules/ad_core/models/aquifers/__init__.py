"""MRST ad-core aquifer model layout."""

from pymrst.modules.ad_core.aquifer import (
    AquiferConnections,
    AquiferModel,
    compute_aquifer_fluxes,
    equilibrium_aquifer_pressure,
    fetkovich_coefficient,
    update_aquifer_state,
)
from pymrst.modules.ad_core.models.aquifers.select_aquifer_from_deck import (
    aquifer_model_from_processed,
    select_aquifer_from_deck,
)

__all__ = [
    "AquiferConnections",
    "AquiferModel",
    "aquifer_model_from_processed",
    "compute_aquifer_fluxes",
    "equilibrium_aquifer_pressure",
    "fetkovich_coefficient",
    "select_aquifer_from_deck",
    "update_aquifer_state",
]

