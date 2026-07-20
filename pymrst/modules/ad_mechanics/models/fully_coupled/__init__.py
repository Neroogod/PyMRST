"""MRST-shaped fully coupled ad-mechanics model landing modules."""

from pymrst.modules.ad_mechanics.models.fully_coupled.mech_black_oil_model import (
    MechBlackOilModel,
)
from pymrst.modules.ad_mechanics.models.fully_coupled.mech_oil_water_model import (
    MechOilWaterModel,
)
from pymrst.modules.ad_mechanics.models.fully_coupled.mech_water_model import (
    MechWaterModel,
)

__all__ = ["MechBlackOilModel", "MechOilWaterModel", "MechWaterModel"]
