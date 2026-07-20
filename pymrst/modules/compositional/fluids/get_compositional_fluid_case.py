"""Compatibility landing file for MRST `getCompositionalFluidCase.m`."""

from pymrst.modules.compositional.fluids.fluid_cases import (
    FluidCaseInfo,
    get_compositional_fluid_case,
)

__all__ = ["FluidCaseInfo", "get_compositional_fluid_case"]
