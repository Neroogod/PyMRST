"""Total thermal energy (MRST ``TotalThermalEnergy.m`` layout)."""

from pymrst.modules.geothermal.geothermal_model import GeothermalModel


def total_thermal_energy(model: GeothermalModel, p, t):
    """Fluid plus rock thermal energy per cell."""
    return model.energy(p, t)


__all__ = ["GeothermalModel", "total_thermal_energy"]
