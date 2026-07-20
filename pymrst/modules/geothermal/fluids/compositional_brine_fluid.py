"""Compositional brine fluid (MRST ``CompositionalBrineFluid.m`` subset)."""

from dataclasses import dataclass

from pymrst.modules.geothermal.eos import density_brine, viscosity_brine
from pymrst.modules.geothermal.geothermal_model import CP_W, LAMBDA_F


@dataclass
class ThermalFluid:
    """Water or brine with Spivey/Kestin density and viscosity."""

    salinity: float = 0.0
    Cp: float = CP_W
    lambda_f: float = LAMBDA_F

    def rho(self, p, t):
        return density_brine(p, t, self.salinity)

    def mu(self, p, t):
        return viscosity_brine(p, t, self.salinity)

    def u(self, p, t):
        return self.Cp * t

    def h(self, p, t):
        return self.u(p, t) + p / self.rho(p, t)


__all__ = ["ThermalFluid"]
