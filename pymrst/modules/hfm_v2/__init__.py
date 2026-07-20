"""MRST ``hfm_v2`` compatibility surface.

The local MRST tree keeps ``hfm`` and ``hfm_v2`` as parallel module layouts.
PyMRST shares the validated EDFM kernels and exposes the newer names here.
"""

from pymrst.modules.hfm import (
    EDFMGrid,
    FractureLine,
    PlanarFracture,
    Projection3D,
    avg_frac_distance,
    build_edfm,
    build_edfm_3d,
    project_planar_fracture_to_grid,
    solve_incomp_edfm,
    solve_incomp_edfm_3d,
)

__all__ = [
    "EDFMGrid", "FractureLine", "PlanarFracture", "Projection3D",
    "avg_frac_distance", "build_edfm", "build_edfm_3d",
    "project_planar_fracture_to_grid", "solve_incomp_edfm",
    "solve_incomp_edfm_3d",
]
