"""Temperature state accessor (MRST ``Temperature.m`` layout)."""

from pymrst.modules.geothermal.geothermal_model import GeothermalState


def temperature(state: GeothermalState):
    """Return the state's temperature field."""
    return state.T


__all__ = ["GeothermalState", "temperature"]
