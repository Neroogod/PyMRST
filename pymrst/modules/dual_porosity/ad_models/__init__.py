"""MRST-shaped dual-porosity AD model landing modules."""

from pymrst.modules.dual_porosity.ad_models.two_phase_oil_water_model_dp import (
    DPOilWaterState,
    TwoPhaseOilWaterModelDP,
)
from pymrst.modules.dual_porosity.ad_models.water_model_dp import (
    DPWaterState,
    WaterModelDP,
)

__all__ = [
    "DPOilWaterState",
    "DPWaterState",
    "TwoPhaseOilWaterModelDP",
    "WaterModelDP",
]
