"""Backward-compatible import path for the CO2-brine water-gas model."""

from pymrst.modules.co2lab_common.co2_brine import (
    CO2BrineModel,
    WaterGasFluid,
    WGState,
    co2_brine_fluid,
)

__all__ = [
    "CO2BrineModel",
    "WaterGasFluid",
    "WGState",
    "co2_brine_fluid",
]
