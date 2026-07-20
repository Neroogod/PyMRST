"""Compatibility landing file for MRST ``MechOilWaterModel.m``."""

from pymrst.modules.ad_mechanics.mech_blackoil import MechBlackOilModel

MechOilWaterModel = MechBlackOilModel

__all__ = ["MechOilWaterModel", "MechBlackOilModel"]
