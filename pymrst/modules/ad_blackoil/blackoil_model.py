"""Legacy alias for MRST-layout `ad_blackoil.models.three_phase_black_oil_model`."""

from pymrst.modules.ad_blackoil.models.three_phase_black_oil_model import (
    BOState,
    ThreePhaseBlackOilModelAD,
    initial_state_spe1,
)

__all__ = ["BOState", "ThreePhaseBlackOilModelAD", "initial_state_spe1"]
