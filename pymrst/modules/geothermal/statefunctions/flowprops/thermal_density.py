"""Thermal fluid density (MRST ``ThermalDensity.m`` layout)."""

from pymrst.modules.geothermal.geothermal_model import ThermalFluid


def thermal_density(fluid: ThermalFluid, p, t):
    """Water/brine density for geothermal flow."""
    return fluid.rho(p, t)


__all__ = ["ThermalFluid", "thermal_density"]
