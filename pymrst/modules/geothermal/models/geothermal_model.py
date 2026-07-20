"""Compatibility landing file for MRST ``GeothermalModel.m``."""

from pymrst.modules.geothermal.geothermal_model import (
    GeothermalModel,
    GeothermalState,
    ThermalFluid,
    ThermalRock,
)

__all__ = ["GeothermalModel", "GeothermalState", "ThermalFluid", "ThermalRock"]
