"""Water-mechanics equations (MRST ``equationsWaterMech.m`` layout)."""

from pymrst.modules.ad_mechanics.poromech import (
    MechWaterModel,
    MechWaterWellModel,
    PoroMechanicalPoreVolume,
)
from pymrst.modules.ad_core.statefunctions.state_function import State
from pymrst.utils.adi import init_variables_adi


def equations_water_mech(model: MechWaterModel | MechWaterWellModel, state0, state, dt):
    """Return flow and mechanics residuals for the fully coupled water model."""
    if hasattr(model, "get_equations"):
        return model.get_equations(state0, state, dt)

    p, xd = init_variables_adi(state["pressure"], state["xd"])
    cur = State(pressure=p, xd=xd)
    flow = model.flow_equation(state0, cur, dt)
    mech = model.mech.equations(xd, p)
    return [flow, mech]


__all__ = [
    "MechWaterModel",
    "MechWaterWellModel",
    "PoroMechanicalPoreVolume",
    "equations_water_mech",
]
