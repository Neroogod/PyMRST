"""Fluid internal energy (MRST ``PhaseInternalEnergy.m`` layout)."""

from pymrst.modules.geothermal.geothermal_model import ThermalFluid


def phase_internal_energy(fluid: ThermalFluid, p, t):
    """Internal energy ``u = Cp*T``."""
    return fluid.Cp * t


__all__ = ["ThermalFluid", "phase_internal_energy"]
