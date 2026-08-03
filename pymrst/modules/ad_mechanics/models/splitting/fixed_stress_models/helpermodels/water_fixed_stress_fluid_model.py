"""Compatibility adapter for MRST ``WaterFixedStressFluidModel.m``."""

from dataclasses import dataclass

from pymrst.modules.ad_mechanics.models.poromechanics import FixedStressPoreVolume
from pymrst.modules.ad_mechanics.models.fully_coupled.helpermodels.fluidhelpermodels.water_fluid_model import WaterFluidModel
from pymrst.modules.ad_mechanics.models.splitting.fixed_stress_models.helpermodels._fixed_stress_fluid import FixedStressFluidMixin


@dataclass
class WaterFixedStressFluidModel(FixedStressFluidMixin, WaterFluidModel):
    """Marker/helper carrying the fixed-stress pore-volume state function."""

    G: object = None
    rock: object = None
    fluid: object = None

    pore_volume_class = FixedStressPoreVolume
    variable_names = ("wellSol", "pressure")


__all__ = ["WaterFixedStressFluidModel"]
