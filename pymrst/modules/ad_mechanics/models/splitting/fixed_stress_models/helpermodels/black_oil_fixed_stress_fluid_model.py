"""Compatibility adapter for MRST ``BlackOilFixedStressFluidModel.m``."""

from dataclasses import dataclass

from pymrst.modules.ad_mechanics.poromech import FixedStressPoreVolume


@dataclass
class BlackOilFixedStressFluidModel:
    """Black-oil fixed-stress helper surface."""

    G: object = None
    rock: object = None
    fluid: object = None
    disgas: bool = False

    pore_volume_class = FixedStressPoreVolume

    def get_all_vars_names(self):
        names = ["pressure", "sW"]
        if self.disgas:
            names.append("rs_or_sG")
        return names

    def getAllVarsNames(self):
        return self.get_all_vars_names()


__all__ = ["BlackOilFixedStressFluidModel"]
