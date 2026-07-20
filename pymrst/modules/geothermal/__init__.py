"""Geothermal flow: water/brine EOS and coupled mass-energy (MRST geothermal/)."""

from pymrst.modules.geothermal.geothermal_model import (
    GeothermalModel,
    GeothermalState,
    ThermalFluid,
    ThermalRock,
)
from pymrst.modules.geothermal.eos import (
    density_brine,
    density_pure_water,
    viscosity_brine,
    viscosity_pure_water,
)

__all__ = [
    "density_pure_water", "density_brine",
    "viscosity_pure_water", "viscosity_brine",
    "GeothermalModel", "GeothermalState", "ThermalFluid", "ThermalRock",
]
