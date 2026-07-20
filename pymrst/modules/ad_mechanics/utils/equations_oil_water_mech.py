"""Oil-water mechanics equations (MRST ``equationsOilWaterMech.m`` layout)."""

from pymrst.modules.ad_mechanics.mech_blackoil import MechBlackOilModel
from pymrst.modules.ad_mechanics.poromech import PoroMechanicalPoreVolume

MechOilWaterModel = MechBlackOilModel


def equations_oil_water_mech(model: MechBlackOilModel, state0, state, dt):
    """Return residuals for the two-phase mechanics model."""
    return model.get_equations(state0, state, dt)


__all__ = [
    "MechOilWaterModel",
    "MechBlackOilModel",
    "PoroMechanicalPoreVolume",
    "equations_oil_water_mech",
]
