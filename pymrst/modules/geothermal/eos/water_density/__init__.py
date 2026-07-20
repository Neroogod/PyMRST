"""MRST geothermal water-density landing modules."""

from pymrst.modules.geothermal.eos.water_density.density_brine import (
    density_brine,
)
from pymrst.modules.geothermal.eos.water_density.density_pure_water import (
    density_pure_water,
)

__all__ = ["density_brine", "density_pure_water"]
