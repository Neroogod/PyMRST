"""Compatibility adapter for MRST ``WaterFluidModel.m``."""

from dataclasses import dataclass

from pymrst.modules.ad_core.statefunctions.pvt_property_functions import water_phase_fluid


@dataclass
class WaterFluidModel:
    """Small adapter carrying the water phase fluid used by mechanics models."""

    G: object
    rock: object
    fluid: object

    @classmethod
    def from_properties(cls, g, rock, b, mu: float, rho: float):
        return cls(g, rock, water_phase_fluid(b, mu, rho))

    def get_all_vars_names(self):
        return ["pressure"]

    def getAllVarsNames(self):  # MRST-style alias
        return self.get_all_vars_names()


__all__ = ["WaterFluidModel"]
