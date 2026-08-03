"""Compatibility adapter for MRST ``WaterFluidModel.m``."""

from dataclasses import dataclass

from pymrst.modules.ad_core.statefunctions.pvt_property_functions import water_phase_fluid
from pymrst.modules.ad_mechanics.models.fully_coupled.helpermodels.fluidhelpermodels._coupled_fluid_helper import CoupledFluidHelper


@dataclass
class WaterFluidModel(CoupledFluidHelper):
    """Small adapter carrying the water phase fluid used by mechanics models."""

    G: object
    rock: object
    fluid: object

    primary_variable_names = ("pressure",)
    variable_names = ("wellSol", "pressure")

    @classmethod
    def from_properties(cls, g, rock, b, mu: float, rho: float):
        return cls(g, rock, water_phase_fluid(b, mu, rho))

__all__ = ["WaterFluidModel"]
