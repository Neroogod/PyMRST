"""Compatibility landing file for MRST `getCompositionalMixtureCase.m`."""

from pymrst.modules.compositional.fluids.fluid_cases import (
    FluidCaseInfo,
    get_compositional_mixture_case,
)

__all__ = ["FluidCaseInfo", "get_compositional_mixture_case"]
