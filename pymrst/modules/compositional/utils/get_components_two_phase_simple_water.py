"""Component phase masses for simple water compositional models.

Port of MRST ``getComponentsTwoPhaseSimpleWater.m``.
"""

from __future__ import annotations


def get_components_two_phase_simple_water(model, rho, sT, xM, yM):
    """Return per-component phase mass entries in MRST cell-array order.

    For water-active models, hydrocarbon components are placed in oil/gas
    phases and the immiscible water component is appended as the final
    component. Without water, each component has liquid/vapour entries only.
    """
    if getattr(model, "water", False):
        components = [[None, rho[1] * x, rho[2] * y] for x, y in zip(xM, yM)]
        components.append([rho[0] * sT, None, None])
        return components
    return [[rho[0] * x, rho[1] * y] for x, y in zip(xM, yM)]


def getComponentsTwoPhaseSimpleWater(model, rho, sT, xM, yM):
    """MRST camelCase alias for :func:`get_components_two_phase_simple_water`."""
    return get_components_two_phase_simple_water(model, rho, sT, xM, yM)


__all__ = ["getComponentsTwoPhaseSimpleWater", "get_components_two_phase_simple_water"]
