"""Shale-specific compositional models and equations."""

from pymrst.modules.shale.compositional_fns.get_shale_comp_fluid_case import ShaleFluidCaseInfo, get_shale_comp_fluid_case
from pymrst.modules.shale.compositional_fns.nat_vars_shale_model import NatVarsShaleModel
from pymrst.modules.shale.compositional_fns.equations_shale_natural_vars import (
    add_sorbed_accumulation,
    apply_gangi_velocity,
    competitive_langmuir_components,
    equations_shale_natural_vars,
    gas_diffusion_flux,
    langmuir_sorbed_mass,
    non_darcy_velocity,
    sorbed_accumulation,
)
from pymrst.modules.shale.compositional_fns.sorbed_compositional_fluid import SorbedCompositionalFluid

__all__ = [
    "NatVarsShaleModel", "ShaleFluidCaseInfo", "SorbedCompositionalFluid",
    "add_sorbed_accumulation", "apply_gangi_velocity",
    "competitive_langmuir_components", "equations_shale_natural_vars",
    "gas_diffusion_flux", "get_shale_comp_fluid_case",
    "langmuir_sorbed_mass", "non_darcy_velocity", "sorbed_accumulation",
]
