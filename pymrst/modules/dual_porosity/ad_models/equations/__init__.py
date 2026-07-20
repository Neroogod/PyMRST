"""MRST-shaped dual-porosity equation landing modules."""

from pymrst.modules.dual_porosity.ad_models.equations.equations_oil_water_dp import (
    DPOilWaterState,
    TwoPhaseOilWaterModelDP,
)
from pymrst.modules.dual_porosity.ad_models.equations.equations_water_dp import (
    DPWaterState,
    WaterModelDP,
)

__all__ = [
    "DPOilWaterState",
    "DPWaterState",
    "TwoPhaseOilWaterModelDP",
    "WaterModelDP",
]
