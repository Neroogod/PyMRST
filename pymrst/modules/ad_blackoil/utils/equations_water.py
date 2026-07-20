"""MRST `ad-blackoil/utils/equationsWater.m`."""

from typing import List

from pymrst.modules.ad_core.utils.boundary_conditions import add_bc_and_sources
from pymrst.modules.ad_core.statefunctions.state_function import State


def equations_water(state0: State, state: State, model, dt: float) -> List:
    """Assemble the single-phase water residual and facility equations."""
    st = model.init_state_ad(state)
    model._conv_state = st
    st0 = model._state0_props(state0)

    eqs = model.conservation_equations(st0, st, dt)
    eqs = add_bc_and_sources(model, eqs, st, model.bc, model.src)

    q_s = [st["qWs"]]
    cq_s, well_eqs, ctrl = model.facility.well_equations(
        model, st, q_s, st["bhp"], state0=st0
    )
    eqs = model.facility.apply_to_reservoir(eqs, cq_s, model._basis_scale())
    return eqs + well_eqs + [ctrl]


__all__ = ["equations_water"]
