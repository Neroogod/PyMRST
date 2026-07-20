"""Port of MRST ``compositional/utils/units/fromFahrenheit.m``."""

from __future__ import annotations

import numpy as np


RANKINE = 5.0 / 9.0


def from_fahrenheit(T):
    """Convert degrees Fahrenheit to Kelvin."""
    return (np.asarray(T, dtype=float) + 459.67) * RANKINE


def fromFahrenheit(T):
    """MRST camelCase alias for :func:`from_fahrenheit`."""
    return from_fahrenheit(T)


__all__ = ["from_fahrenheit", "fromFahrenheit", "RANKINE"]
