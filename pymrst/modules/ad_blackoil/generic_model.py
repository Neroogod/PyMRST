"""Legacy alias for MRST-layout `ad_blackoil.models.generic_black_oil_model`."""

from pymrst.modules.ad_blackoil.models.generic_black_oil_model import (
    GenericBlackOilModel,
    blackoil_status,
    initial_state_from_deck,
)

__all__ = [
    "GenericBlackOilModel",
    "blackoil_status",
    "initial_state_from_deck",
]
