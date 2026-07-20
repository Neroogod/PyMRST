"""Compatibility adapter for MRST ``OilWaterFluidModel.m``."""

from dataclasses import dataclass

from pymrst.modules.ad_core.statefunctions.pvt_property_functions import simple_phase_fluid


@dataclass
class OilWaterFluidModel:
    """Two-phase oil-water helper surface used by ``MechOilWaterModel``."""

    G: object
    rock: object
    fluid: object

    @classmethod
    def from_fluid(cls, g, rock, fluid):
        return cls(g, rock, simple_phase_fluid(fluid))

    def get_all_vars_names(self):
        return ["pressure", "sW"]

    def getAllVarsNames(self):
        return self.get_all_vars_names()


__all__ = ["OilWaterFluidModel"]
