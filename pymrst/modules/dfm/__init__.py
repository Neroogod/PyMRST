"""Discrete fracture model compatibility helpers.

This module exposes a conforming-workflow facade over PyMRST's EDFM kernels.
For now, 2D line fractures and 3D planar fractures share the same validated
fracture-cell/NNC pressure workflow as ``hfm``/``hfm_v2``.
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


def build_dfm(*args, **kwargs):
    """Build a 2D DFM-compatible fracture workflow."""
    return build_edfm(*args, **kwargs)


def build_dfm_3d(*args, **kwargs):
    """Build a 3D DFM-compatible planar fracture workflow."""
    return build_edfm_3d(*args, **kwargs)


__all__ = [
    "EDFMGrid", "FractureLine", "PlanarFracture", "build_dfm",
    "build_dfm_3d", "solve_incomp_edfm", "solve_incomp_edfm_3d",
]
