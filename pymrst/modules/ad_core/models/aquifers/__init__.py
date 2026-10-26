"""MRST ad-core aquifer model layout (`ad-core/models/aquifers/`)."""

from pymrst.modules.ad_core.models.aquifers.aquifer import (
    AquiferConnections,
    AquiferModelAD,
    compute_aquifer_fluxes,
    equilibrium_aquifer_pressure,
    fetkovich_coefficient,
    update_aquifer_state,
)
from pymrst.modules.ad_core.models.aquifers.aquifer_model import (
    AquiferModel,
    add_aquifers_contribution,
)
from pymrst.modules.ad_core.models.aquifers.carter_tracy_aquifer import (
    CarterTracyAquiferModel,
    van_everdingen_hurst_pd,
)
from pymrst.modules.ad_core.models.aquifers.select_aquifer_from_deck import (
    aquifer_model_from_processed,
    select_aquifer_from_deck,
    select_aquifer_from_deck_ad,
)

__all__ = [
    "AquiferConnections",
    "AquiferModel",
    "AquiferModelAD",
    "add_aquifers_contribution",
    "CarterTracyAquiferModel",
    "aquifer_model_from_processed",
    "compute_aquifer_fluxes",
    "equilibrium_aquifer_pressure",
    "fetkovich_coefficient",
    "select_aquifer_from_deck",
    "select_aquifer_from_deck_ad",
    "update_aquifer_state",
    "van_everdingen_hurst_pd",
]
