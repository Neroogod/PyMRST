"""Phase enthalpy (MRST ``PhaseEnthalpy.m`` layout)."""

from pymrst.modules.geothermal.geothermal_model import ThermalFluid


def phase_enthalpy(fluid: ThermalFluid, p, t):
    """Single geothermal phase enthalpy."""
    return fluid.u(p, t) + p / fluid.rho(p, t)


__all__ = ["ThermalFluid", "phase_enthalpy"]
