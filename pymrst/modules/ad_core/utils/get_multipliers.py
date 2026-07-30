"""Evaluate pressure-dependent reservoir multipliers."""

from __future__ import annotations


def _field(obj, name):
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def get_multipliers(fluid, pressure, previous_pressure):
    """Return current and previous PV, transmissibility, and mobility factors."""
    pv_function = _field(fluid, "pvMultR")
    mobility_function = _field(fluid, "tranMultR")
    transmissibility_function = _field(fluid, "transMult")
    pv = pv_function(pressure) if pv_function is not None else 1.0
    transmissibility = (
        transmissibility_function(pressure)
        if transmissibility_function is not None
        else 1.0
    )
    mobility = (
        mobility_function(pressure) if mobility_function is not None else 1.0
    )
    pv_previous = (
        pv_function(previous_pressure) if pv_function is not None else 1.0
    )
    transmissibility_previous = (
        transmissibility_function(previous_pressure)
        if transmissibility_function is not None
        else 1.0
    )
    mobility_previous = (
        mobility_function(previous_pressure)
        if mobility_function is not None
        else 1.0
    )
    return (
        pv,
        transmissibility,
        mobility,
        pv_previous,
        transmissibility_previous,
        mobility_previous,
    )


getMultipliers = get_multipliers

__all__ = ["get_multipliers", "getMultipliers"]
