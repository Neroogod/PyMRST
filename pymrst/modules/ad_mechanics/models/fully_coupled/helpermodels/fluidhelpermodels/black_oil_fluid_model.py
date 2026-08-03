"""Compatibility adapter for MRST ``BlackOilFluidModel.m``."""

from dataclasses import dataclass

from pymrst.modules.ad_core.statefunctions.pvt_property_functions import blackoil_phase_fluid
from pymrst.modules.ad_mechanics.models.fully_coupled.helpermodels.fluidhelpermodels._coupled_fluid_helper import CoupledFluidHelper


@dataclass
class BlackOilFluidModel(CoupledFluidHelper):
    """Black-oil helper surface for MRST-shaped mechanics imports."""

    G: object
    rock: object
    fluid: object
    disgas: bool = False
    vapoil: bool = False

    primary_variable_names = ("pressure", "sW", "x")
    variable_names = (
        "wellSol", "pressure", "s", "rs", "rv", "sW", "sG", "sO",
        "water", "oil", "gas",
    )

    @classmethod
    def from_raw_fluid(cls, g, rock, raw_fluid, disgas=False, vapoil=False):
        return cls(g, rock, blackoil_phase_fluid(raw_fluid, vapoil=vapoil),
                   disgas=disgas, vapoil=vapoil)

__all__ = ["BlackOilFluidModel"]
