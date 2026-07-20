"""Backward-compatible import path for the Carter-Tracy aquifer model.

The implementation lives in `ad_core.models.aquifers.carter_tracy_aquifer`,
alongside the Fetkovich `AquiferModel` in MRST's `ad-core/models/aquifers/`.
"""

from pymrst.modules.ad_core.models.aquifers.carter_tracy_aquifer import (
    CarterTracyAquiferModel,
    van_everdingen_hurst_pd,
)

__all__ = ["CarterTracyAquiferModel", "van_everdingen_hurst_pd"]
