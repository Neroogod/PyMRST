"""Port of MRST ``compositional/utils/units/fromCelsius.m``."""

from __future__ import annotations

import numpy as np


def from_celsius(T):
    """Convert degrees Celsius to Kelvin."""
    return np.asarray(T, dtype=float) + 273.15


def fromCelsius(T):
    """MRST camelCase alias for :func:`from_celsius`."""
    return from_celsius(T)


__all__ = ["from_celsius", "fromCelsius"]
