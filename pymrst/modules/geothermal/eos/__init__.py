"""MRST-shaped geothermal EOS landing modules."""

from pymrst.modules.geothermal.eos.water_density import (
    density_brine,
    density_pure_water,
)
from pymrst.modules.geothermal.eos.water_viscosity import (
    viscosity_brine,
    viscosity_pure_water,
)

__all__ = [
    "density_brine",
    "density_pure_water",
    "viscosity_brine",
    "viscosity_pure_water",
]
