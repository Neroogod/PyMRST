"""Compatibility landing file for MRST `getBenchmarkMixture.m`."""

from pymrst.modules.compositional.fluids.fluid_cases import (
    FluidCaseInfo,
    get_benchmark_mixture,
)

__all__ = ["FluidCaseInfo", "get_benchmark_mixture"]
