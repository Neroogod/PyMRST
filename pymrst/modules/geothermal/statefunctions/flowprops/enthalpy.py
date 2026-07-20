"""Fluid enthalpy (MRST ``Enthalpy.m`` layout)."""

from pymrst.modules.geothermal.geothermal_model import ThermalFluid


def enthalpy(fluid: ThermalFluid, p, t):
    """Fluid enthalpy ``h = Cp*T + p/rho``."""
    return fluid.u(p, t) + p / fluid.rho(p, t)


__all__ = ["ThermalFluid", "enthalpy"]
