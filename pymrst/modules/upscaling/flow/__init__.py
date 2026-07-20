"""MRST upscaling compatibility landing modules."""

from pymrst.modules.upscaling.flow.upscale_perm import upscale_perm
from pymrst.modules.upscaling.flow.upscale_permeability_fixed import (
    upscale_perm_fixed,
    upscale_permeability_fixed,
)
from pymrst.modules.upscaling.flow.upscale_poro import upscale_poro
from pymrst.modules.upscaling.flow.upscale_rock import upscale_rock

__all__ = [
    "upscale_perm",
    "upscale_perm_fixed",
    "upscale_permeability_fixed",
    "upscale_poro",
    "upscale_rock",
]
