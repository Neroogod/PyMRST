"""Shale fracture-workflow compatibility helpers.

MRST's shale module layers shale-specific examples and pEDFM helpers on top of
HFM-style fracture grids. PyMRST exposes the shared fracture workflow here so
shale examples can use the same 2D/3D EDFM kernels.
"""

from pymrst.modules.hfm import (
    EDFMGrid,
    FractureLine,
    PlanarFracture,
    build_edfm,
    build_edfm_3d,
    solve_incomp_edfm,
    solve_incomp_edfm_3d,
)


def build_shale_edfm(*args, **kwargs):
    """Build a shale pEDFM-compatible 2D fracture grid."""
    return build_edfm(*args, **kwargs)


def build_shale_edfm_3d(*args, **kwargs):
    """Build a shale pEDFM-compatible 3D planar fracture grid."""
    return build_edfm_3d(*args, **kwargs)


__all__ = [
    "EDFMGrid", "FractureLine", "PlanarFracture", "build_shale_edfm",
    "build_shale_edfm_3d", "solve_incomp_edfm", "solve_incomp_edfm_3d",
]
