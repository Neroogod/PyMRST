"""Compatibility adapter for MRST ``WaterFixedStressFluidModel.m``."""

from dataclasses import dataclass

from pymrst.modules.ad_mechanics.poromech import FixedStressPoreVolume


@dataclass
class WaterFixedStressFluidModel:
    """Marker/helper carrying the fixed-stress pore-volume state function."""

    G: object = None
    rock: object = None
    fluid: object = None

    pore_volume_class = FixedStressPoreVolume

    def get_all_vars_names(self):
        return ["pressure"]

    def getAllVarsNames(self):
        return self.get_all_vars_names()


__all__ = ["WaterFixedStressFluidModel"]
