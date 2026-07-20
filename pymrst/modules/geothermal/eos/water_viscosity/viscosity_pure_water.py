"""Pure-water viscosity correlation (MRST ``viscosity_pure_water.m`` layout)."""

from pymrst.modules.geothermal.water_props import Num
from pymrst.modules.geothermal.eos.water_viscosity.viscosity_brine import (
    viscosity_brine,
)


def viscosity_pure_water(p: Num, t: Num) -> Num:
    """Pure-water viscosity in Pa s."""
    return viscosity_brine(p, t, 0.0)


__all__ = ["viscosity_pure_water"]
