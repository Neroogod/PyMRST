"""Black-oil mechanics equations (MRST ``equationsBlackOilMech.m`` layout)."""

from pymrst.modules.ad_mechanics.mech_blackoil import MechBlackOilModel
from pymrst.modules.ad_mechanics.poromech import PoroMechanicalPoreVolume


def equations_black_oil_mech(model: MechBlackOilModel, state0, state, dt):
    """Return reservoir, well, control, and mechanics residuals."""
    return model.get_equations(state0, state, dt)


__all__ = [
    "MechBlackOilModel",
    "PoroMechanicalPoreVolume",
    "equations_black_oil_mech",
]
