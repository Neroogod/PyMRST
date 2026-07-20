"""Total heat flux (MRST ``HeatFlux.m`` layout)."""

from pymrst.modules.geothermal.geothermal_model import GeothermalModel
from pymrst.modules.geothermal.statefunctions.flux.advective_heat_flux import (
    advective_heat_flux,
)
from pymrst.modules.geothermal.statefunctions.flux.conductive_heat_flux import (
    conductive_heat_flux,
)


def heat_flux(model: GeothermalModel, p, t):
    """Return advective plus conductive interior heat flux."""
    return advective_heat_flux(model, p, t) + conductive_heat_flux(model, t)


__all__ = ["GeothermalModel", "heat_flux"]
