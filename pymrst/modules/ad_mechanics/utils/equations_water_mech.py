"""Water-mechanics equations (MRST ``equationsWaterMech.m`` layout)."""

from pymrst.modules.ad_mechanics.models.poromechanics import (
    MechWaterModel,
    MechWaterWellModel,
    PoroMechanicalPoreVolume,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.utils.adi import init_variables_adi
from pymrst.modules.ad_mechanics.utils._coupled_equation_contract import format_equation_output


def equations_water_mech(model: MechWaterModel | MechWaterWellModel, state0,
                         state, dt, driving_forces=None, *,
                         return_metadata=False, iteration=-1,
                         res_only=False, **kwargs):
    """Assemble water/mechanics equations.

    ``driving_forces``, ``iteration`` and ``res_only`` mirror the translated
    MRST entry point.  Forces are already owned by PyMRST's model/facility;
    accepting them here keeps schedule-driven call sites interoperable.
    Set ``return_metadata=True`` for ``(eqs, names, types, state)``.
    """
    if hasattr(model, "get_equations"):
        eqs = model.get_equations(state0, state, dt)
        return format_equation_output(model, state, eqs, ("water",),
                                      return_metadata)

    p, xd = init_variables_adi(state["pressure"], state["xd"])
    cur = State(pressure=p, xd=xd)
    flow = model.flow_equation(state0, cur, dt)
    mech = model.mech.equations(xd, p)
    eqs = [flow, mech]
    return format_equation_output(model, state, eqs, ("water",),
                                  return_metadata)


__all__ = [
    "MechWaterModel",
    "MechWaterWellModel",
    "PoroMechanicalPoreVolume",
    "equations_water_mech",
]
