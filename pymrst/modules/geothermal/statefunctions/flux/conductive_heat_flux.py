"""Conductive heat flux (MRST ``ConductiveHeatFlux.m`` layout)."""

from pymrst.modules.geothermal.geothermal_model import GeothermalModel


def conductive_heat_flux(model: GeothermalModel, temperature):
    """Return interior conductive heat flux ``-T_heat * grad(T)``."""
    return -model.T_heat * model.ops.grad(temperature)


__all__ = ["GeothermalModel", "conductive_heat_flux"]
