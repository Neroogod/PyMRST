"""MRST-compatible nozzle valve pressure-drop model."""

from __future__ import annotations

import numpy as np

from pymrst.utils.adi import value


def nozzle_valve(v, rho, diameter, discharge_coeff, flowtype="massRate"):
    """Compute pressure drop across a nozzle valve.

    ``v`` is interpreted as velocity, volume rate, or mass rate according to
    ``flowtype``.  The pressure drop opposes the direction of flow.
    """
    diameter = np.asarray(diameter, dtype=float)
    discharge_coeff = np.asarray(discharge_coeff, dtype=float)
    area = np.pi * (diameter / 2.0) ** 2
    if flowtype == "velocity":
        velocity = v
    elif flowtype == "volumeRate":
        velocity = v / area
    elif flowtype == "massRate":
        velocity = v / (rho * area)
    else:
        raise ValueError(f"unknown flow type: {flowtype}")
    return -(np.sign(value(velocity)) * rho * velocity**2) / (
        2.0 * discharge_coeff**2
    )


nozzleValve = nozzle_valve

__all__ = ["nozzle_valve", "nozzleValve"]
