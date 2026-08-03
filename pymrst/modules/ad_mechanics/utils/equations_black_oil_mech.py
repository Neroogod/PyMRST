"""Black-oil mechanics equations (MRST ``equationsBlackOilMech.m`` layout)."""

from pymrst.modules.ad_mechanics.models.fully_coupled.mech_black_oil_model import MechBlackOilModel
from pymrst.modules.ad_mechanics.models.poromechanics import PoroMechanicalPoreVolume
from pymrst.modules.ad_mechanics.utils._coupled_equation_contract import format_equation_output


def equations_black_oil_mech(model: MechBlackOilModel, state0, state, dt,
                             driving_forces=None, *, return_metadata=False,
                             iteration=-1, res_only=False, verbose=False,
                             **kwargs):
    """Assemble reservoir, well, control, and mechanics residuals."""
    eqs = model.get_equations(state0, state, dt)
    phases = tuple(getattr(model, "phase_names", ("water", "oil", "gas")))
    phases = phases[: getattr(model, "n_phases", len(phases))]
    return format_equation_output(model, state, eqs, phases, return_metadata)


__all__ = [
    "MechBlackOilModel",
    "PoroMechanicalPoreVolume",
    "equations_black_oil_mech",
]
