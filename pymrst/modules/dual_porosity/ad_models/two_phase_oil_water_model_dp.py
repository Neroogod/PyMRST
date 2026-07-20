"""Compatibility landing file for MRST ``TwoPhaseOilWaterModelDP.m``."""

from pymrst.modules.dual_porosity.dp_model import (
    DPOilWaterState,
    TwoPhaseOilWaterModelDP,
)

__all__ = ["DPOilWaterState", "TwoPhaseOilWaterModelDP"]
