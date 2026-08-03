"""Compatibility adapter for MRST ``OilWaterFluidModel.m``."""

from dataclasses import dataclass

from pymrst.modules.ad_core.statefunctions.pvt_property_functions import simple_phase_fluid
from pymrst.modules.ad_mechanics.models.fully_coupled.helpermodels.fluidhelpermodels._coupled_fluid_helper import CoupledFluidHelper


@dataclass
class OilWaterFluidModel(CoupledFluidHelper):
    """Two-phase oil-water helper surface used by ``MechOilWaterModel``."""

    G: object
    rock: object
    fluid: object

    primary_variable_names = ("pressure", "sW")
    variable_names = ("wellSol", "pressure", "s", "sW", "sO", "water", "oil")

    @classmethod
    def from_fluid(cls, g, rock, fluid):
        return cls(g, rock, simple_phase_fluid(fluid))

__all__ = ["OilWaterFluidModel"]
