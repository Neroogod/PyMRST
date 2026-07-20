"""Backward-compatible import path for vertical-equilibrium fluid helpers."""

from pymrst.modules.co2lab_ve.ve_fluids import (
    VECapillaryFringeFluid,
    VECapLinearFluid,
    VETable,
    free_sg,
    make_ve_table_h,
    make_ve_table_p,
    upper_cap_pressure,
)

__all__ = [
    "VECapillaryFringeFluid",
    "VECapLinearFluid",
    "VETable",
    "free_sg",
    "make_ve_table_h",
    "make_ve_table_p",
    "upper_cap_pressure",
]
