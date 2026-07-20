"""Compatibility adapter for MRST ``BlackOilFluidModel.m``."""

from dataclasses import dataclass

from pymrst.modules.ad_core.statefunctions.pvt_property_functions import blackoil_phase_fluid


@dataclass
class BlackOilFluidModel:
    """Black-oil helper surface for MRST-shaped mechanics imports."""

    G: object
    rock: object
    fluid: object
    disgas: bool = False
    vapoil: bool = False

    @classmethod
    def from_raw_fluid(cls, g, rock, raw_fluid, disgas=False, vapoil=False):
        return cls(g, rock, blackoil_phase_fluid(raw_fluid, vapoil=vapoil),
                   disgas=disgas, vapoil=vapoil)

    def get_all_vars_names(self):
        names = ["pressure", "sW"]
        if self.disgas or self.vapoil:
            names.append("rs_or_sG")
        return names

    def getAllVarsNames(self):
        return self.get_all_vars_names()


__all__ = ["BlackOilFluidModel"]
