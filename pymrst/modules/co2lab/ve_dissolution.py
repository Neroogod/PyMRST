"""Backward-compatible import path for VE dissolution helpers."""

from pymrst.modules.co2lab_ve.ve_dissolution import (
    DissolutionFluid,
    VEDisState,
    VEDissolutionModel,
)

__all__ = [
    "DissolutionFluid",
    "VEDisState",
    "VEDissolutionModel",
]
