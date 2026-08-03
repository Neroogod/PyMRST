"""Compatibility adapter for MRST ``OilWaterFixedStressFluidModel.m``."""

from dataclasses import dataclass

from pymrst.modules.ad_mechanics.models.poromechanics import FixedStressPoreVolume
from pymrst.modules.ad_mechanics.models.fully_coupled.helpermodels.fluidhelpermodels.oil_water_fluid_model import OilWaterFluidModel
from pymrst.modules.ad_mechanics.models.splitting.fixed_stress_models.helpermodels._fixed_stress_fluid import FixedStressFluidMixin


@dataclass
class OilWaterFixedStressFluidModel(FixedStressFluidMixin, OilWaterFluidModel):
    """Two-phase fixed-stress helper surface."""

    G: object = None
    rock: object = None
    fluid: object = None

    pore_volume_class = FixedStressPoreVolume
    variable_names = ("wellSol", "pressure", "s")


__all__ = ["OilWaterFixedStressFluidModel"]
