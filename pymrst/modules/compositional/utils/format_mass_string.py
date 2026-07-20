"""Human-readable mass formatter (MRST ``formatMassString.m``)."""

from __future__ import annotations

import math


_MASS_UNITS = (
    (1.0e9, "Megatonne"),
    (1.0e3, "Tonne"),
    (1.0, "Kilogram"),
    (1.0e-3, "Gram"),
)


def format_mass_string(mass: float, limit: float = math.inf) -> str:
    """Return a human-readable mass string using MRST's unit order."""
    mass = float(mass)
    if mass > 0.0:
        out = ""
    elif mass < 0.0:
        out = "-"
    else:
        return "0 Kilogram"

    remaining = abs(mass)
    added = False
    count = 0
    for scale, name in _MASS_UNITS:
        amount = math.floor(remaining / scale)
        if amount <= 0:
            continue
        space = ", " if added else ""
        count += int(added)
        plural = "" if amount == 1 else "s"
        if count == limit:
            out += f"{space}{remaining / scale:1.2f} {name}{plural}"
            break
        out += f"{space}{amount:d} {name}{plural}"
        remaining -= amount * scale
        added = True
    return out


def formatMassString(mass: float, limit: float = math.inf) -> str:
    """MRST camelCase alias for :func:`format_mass_string`."""
    return format_mass_string(mass, limit)


__all__ = ["formatMassString", "format_mass_string"]
