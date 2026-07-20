"""MRST-shaped fluid helper model adapters for ``ad-mechanics``."""

from pymrst.modules.ad_mechanics.models.fully_coupled.helpermodels.fluidhelpermodels.black_oil_fluid_model import (
    BlackOilFluidModel,
)
from pymrst.modules.ad_mechanics.models.fully_coupled.helpermodels.fluidhelpermodels.oil_water_fluid_model import (
    OilWaterFluidModel,
)
from pymrst.modules.ad_mechanics.models.fully_coupled.helpermodels.fluidhelpermodels.water_fluid_model import (
    WaterFluidModel,
)

__all__ = ["WaterFluidModel", "OilWaterFluidModel", "BlackOilFluidModel"]
