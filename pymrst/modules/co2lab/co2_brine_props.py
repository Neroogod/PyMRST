"""Backward-compatible import path for CO2-brine thermophysical correlations."""

from pymrst.modules.co2lab_mit.co2_brine_props import (
    brine_co2_viscosity,
    brine_density_rc1970,
    co2_activity_ds2003,
    co2_density,
    co2_density_rk1949,
    co2_viscosity,
    co2_viscosity_f1998,
    gas_mixture_viscosity,
)

__all__ = [
    "brine_co2_viscosity",
    "brine_density_rc1970",
    "co2_activity_ds2003",
    "co2_density",
    "co2_density_rk1949",
    "co2_viscosity",
    "co2_viscosity_f1998",
    "gas_mixture_viscosity",
]
