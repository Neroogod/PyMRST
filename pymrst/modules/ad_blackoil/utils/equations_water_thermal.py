"""MRST ``equationsWaterThermal.m`` compatibility entry point."""

import numpy as np

from pymrst.utils.adi import init_variables_adi


def equations_water_thermal(
    state0, state, model, dt: float, driving_forces=None,
    *, reverse_mode: bool = False, res_only: bool = False, **options,
):
    """Assemble coupled single-phase water mass and energy equations.

    The numerical owner is PyMRST's geothermal kernel. Boundary
    pressure/temperature data, direct sources and wells are all configured on
    the *model*, because that is where the kernel reads them and where the
    injection temperature each of them needs is validated at construction.

    Sources and wells were refused here until the kernel carried them; it now
    does, with the injected enthalpy upwinded exactly as the boundary block
    does it.
    """
    forces = driving_forces or {}
    # Sources and wells used to be refused here. The geothermal kernel now
    # carries both - a prescribed cell rate, and a Peaceman well rate on
    # bottom-hole-pressure control - each with the injected enthalpy upwinded
    # the same way the boundary block does it: water entering the reservoir
    # arrives at the *injection* temperature, water leaving takes the cell's.
    #
    # They are attached to the model rather than passed through here, because
    # that is where the kernel reads them and where the injection temperature
    # each needs is validated at construction.
    for name, attribute in (("src", "src"), ("W", "wells")):
        supplied = forces.get(name)
        if supplied is not None and len(np.atleast_1d(supplied)):
            if not getattr(model, attribute, None):
                raise ValueError(
                    f"driving force {name!r} was passed to the equations but "
                    f"the model carries no {attribute!r}; the geothermal "
                    f"kernel reads them from the model, where their injection "
                    f"temperature is validated - set them on the model")
    if reverse_mode:
        p0, t0 = init_variables_adi(state0.pressure, state0.T)
        numeric_state = state.copy()
        numeric_state.pressure = p0
        numeric_state.T = t0
        return model.assemble(numeric_state, state.pressure, state.T, dt), state
    p, temperature = (state.pressure, state.T) if res_only else init_variables_adi(
        state.pressure, state.T
    )
    return model.assemble(state0, p, temperature, dt), state


equationsWaterThermal = equations_water_thermal

__all__ = ["equations_water_thermal", "equationsWaterThermal"]
