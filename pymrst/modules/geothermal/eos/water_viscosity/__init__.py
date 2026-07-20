"""MRST geothermal water-viscosity landing modules."""

from pymrst.modules.geothermal.eos.water_viscosity.viscosity_brine import (
    viscosity_brine,
)
from pymrst.modules.geothermal.eos.water_viscosity.viscosity_pure_water import (
    viscosity_pure_water,
)

__all__ = ["viscosity_brine", "viscosity_pure_water"]
