"""Advective heat flux (MRST ``AdvectiveHeatFlux.m`` layout)."""

from pymrst.modules.geothermal.geothermal_model import GeothermalModel
from pymrst.utils.adi import value


def advective_heat_flux(model: GeothermalModel, p, t):
    """Return face advective heat flux ``upw(h) * rho*v`` on interior faces."""
    f, ops = model.fluid, model.ops
    rho = f.rho(p, t)
    mob = 1.0 / f.mu(p, t)
    h = f.h(p, t)
    pot = ops.grad(p)
    flag = value(pot) <= 0
    v = -ops.T * ops.upw(flag, mob) * pot
    mass_flux = ops.upw(flag, rho) * v
    return ops.upw(flag, h) * mass_flux


__all__ = ["GeothermalModel", "advective_heat_flux"]
