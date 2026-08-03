"""Numerical benchmark adapters registered by the validation laboratory."""

from .foundation import (
    ACCUMULATION_FLUX_SPEC,
    BUCKLEY_LEVERETT_SPEC,
    DIFFUSION_SPEC,
    ELASTICITY_PATCH_SPEC,
    FLASH_EQUILIBRIUM_SPEC,
    HYDROSTATIC_SPEC,
    MANDEL_SPEC,
    RADIAL_SPEC,
    TERZAGHI_SPEC,
    buckley_leverett_exact,
    diffusion_exact,
    foundation_analytic_registry,
    radial_bessel_exact,
    run_accumulation_flux_benchmark,
    run_buckley_leverett_benchmark,
    run_diffusion_benchmark,
    run_elasticity_patch_benchmark,
    run_flash_equilibrium_benchmark,
    run_hydrostatic_benchmark,
    run_mandel_benchmark,
    run_radial_pressure_benchmark,
    run_terzaghi_benchmark,
)

__all__ = [name for name in globals() if name.isupper() or name.startswith("run_")]
__all__ += ["buckley_leverett_exact", "diffusion_exact",
            "foundation_analytic_registry", "radial_bessel_exact"]
