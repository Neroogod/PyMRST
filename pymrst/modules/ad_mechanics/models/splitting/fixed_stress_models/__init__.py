"""MRST-shaped fixed-stress splitting landing modules."""

from pymrst.modules.ad_mechanics.models.splitting.fixed_stress_models.mech_fluid_fixed_stress_split_model import (
    FixedStressSplitModel,
)
from pymrst.modules.ad_mechanics.models.splitting.fixed_stress_models.helpermodels import (
    BlackOilFixedStressFluidModel,
    OilWaterFixedStressFluidModel,
    WaterFixedStressFluidModel,
)

__all__ = [
    "FixedStressSplitModel",
    "WaterFixedStressFluidModel",
    "OilWaterFixedStressFluidModel",
    "BlackOilFixedStressFluidModel",
]
