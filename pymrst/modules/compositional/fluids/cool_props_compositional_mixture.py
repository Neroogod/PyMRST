"""Compatibility alias for MRST ``CoolPropsCompositionalMixture.m``."""

from pymrst.modules.compositional.fluids.coolprops_compositional_mixture import (
    CoolPropsCompositionalMixture,
    coolprops_compositional_mixture,
)

cool_props_compositional_mixture = coolprops_compositional_mixture

__all__ = [
    "CoolPropsCompositionalMixture",
    "coolprops_compositional_mixture",
    "cool_props_compositional_mixture",
]
