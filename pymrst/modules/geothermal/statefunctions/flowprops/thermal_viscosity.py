"""Thermal fluid viscosity (MRST ``ThermalViscosity.m`` layout)."""

from pymrst.modules.geothermal.geothermal_model import ThermalFluid


def thermal_viscosity(fluid: ThermalFluid, p, t):
    """Water/brine viscosity for geothermal flow."""
    return fluid.mu(p, t)


__all__ = ["ThermalFluid", "thermal_viscosity"]
