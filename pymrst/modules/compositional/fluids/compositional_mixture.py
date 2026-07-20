"""Compatibility landing file for MRST `CompositionalMixture.m`."""

from pymrst.modules.compositional.eos import (
    FlashResult,
    Mixture,
    flash,
    fugacity_coefficients,
    rachford_rice,
    simple_mixture,
    wilson_k,
)
from pymrst.modules.compositional.fluids.fluid_cases import (
    FluidCaseInfo,
    get_benchmark_mixture,
    get_compositional_fluid_case,
    get_compositional_mixture_case,
)

__all__ = [
    "Mixture", "simple_mixture", "wilson_k", "rachford_rice",
    "fugacity_coefficients", "flash", "FlashResult",
    "FluidCaseInfo", "get_benchmark_mixture", "get_compositional_fluid_case",
    "get_compositional_mixture_case",
]
