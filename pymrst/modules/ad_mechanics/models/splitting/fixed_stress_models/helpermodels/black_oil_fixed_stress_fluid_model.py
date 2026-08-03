"""Compatibility adapter for MRST ``BlackOilFixedStressFluidModel.m``."""

from dataclasses import dataclass

from pymrst.modules.ad_mechanics.models.poromechanics import FixedStressPoreVolume
from pymrst.modules.ad_mechanics.models.fully_coupled.helpermodels.fluidhelpermodels.black_oil_fluid_model import BlackOilFluidModel
from pymrst.modules.ad_mechanics.models.splitting.fixed_stress_models.helpermodels._fixed_stress_fluid import FixedStressFluidMixin


@dataclass
class BlackOilFixedStressFluidModel(FixedStressFluidMixin, BlackOilFluidModel):
    """Black-oil fixed-stress helper surface."""

    G: object = None
    rock: object = None
    fluid: object = None
    disgas: bool = False

    pore_volume_class = FixedStressPoreVolume
    variable_names = ("wellSol", "pressure", "s", "rs", "rv")


__all__ = ["BlackOilFixedStressFluidModel"]
