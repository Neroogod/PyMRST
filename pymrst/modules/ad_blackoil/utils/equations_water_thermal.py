"""MRST ``equationsWaterThermal.m`` compatibility entry point."""

from pymrst.utils.adi import init_variables_adi


def equations_water_thermal(
    state0, state, model, dt: float, driving_forces=None,
    *, reverse_mode: bool = False, res_only: bool = False, **options,
):
    """Assemble coupled single-phase water mass and energy equations.

    The numerical owner is PyMRST's validated geothermal kernel. Boundary
    pressure/temperature data are configured on ``WaterThermalModel``; direct
    sources and wells remain unsupported by that kernel and are rejected
    explicitly instead of being silently omitted.
    """
    forces = driving_forces or {}
    if forces.get("src") is not None:
        raise NotImplementedError("thermal-water source terms are not implemented")
    if forces.get("W") is not None and len(forces["W"]):
        raise NotImplementedError("thermal-water well equations are not implemented")
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
