"""Oil-water mechanics equations (MRST ``equationsOilWaterMech.m`` layout)."""

from pymrst.modules.ad_mechanics.models.fully_coupled.mech_black_oil_model import MechBlackOilModel
from pymrst.modules.ad_mechanics.models.fully_coupled.mech_oil_water_model import MechOilWaterModel
from pymrst.modules.ad_mechanics.models.poromechanics import PoroMechanicalPoreVolume
from pymrst.modules.ad_mechanics.utils._coupled_equation_contract import format_equation_output


def equations_oil_water_mech(model: MechOilWaterModel, state0, state, dt,
                             driving_forces=None, *, return_metadata=False,
                             iteration=-1, res_only=False, **kwargs):
    """Assemble distinct two-phase oil-water/mechanics equations."""
    eqs = model.get_equations(state0, state, dt)
    return format_equation_output(model, state, eqs, ("water", "oil"),
                                  return_metadata)


__all__ = [
    "MechOilWaterModel",
    "MechBlackOilModel",
    "PoroMechanicalPoreVolume",
    "equations_oil_water_mech",
]
