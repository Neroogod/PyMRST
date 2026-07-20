"""Compatibility adapter for MRST ``OilWaterFixedStressFluidModel.m``."""

from dataclasses import dataclass

from pymrst.modules.ad_mechanics.poromech import FixedStressPoreVolume


@dataclass
class OilWaterFixedStressFluidModel:
    """Two-phase fixed-stress helper surface."""

    G: object = None
    rock: object = None
    fluid: object = None

    pore_volume_class = FixedStressPoreVolume

    def get_all_vars_names(self):
        return ["pressure", "sW"]

    def getAllVarsNames(self):
        return self.get_all_vars_names()


__all__ = ["OilWaterFixedStressFluidModel"]
