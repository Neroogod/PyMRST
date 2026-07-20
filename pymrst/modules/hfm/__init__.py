"""Hierarchical/embedded discrete fracture models (MRST modules/hfm)."""

from pymrst.modules.hfm.edfm2d import (
    EDFMGrid,
    FractureLine,
    avg_frac_distance,
    build_edfm,
    solve_incomp_edfm,
)
from pymrst.modules.hfm.edfm3d import (
    PlanarFracture,
    Projection3D,
    build_edfm_3d,
    project_planar_fracture_to_grid,
    solve_incomp_edfm_3d,
)
from pymrst.modules.hfm.mechanics import (
    EDFMMechanicsResult,
    FracturePropagationResult,
    build_tedfm_wrapper,
    fracture_tip_mode_i_indicator,
    fracture_normal_strain,
    fracture_normals,
    propagate_fractures_by_aperture,
    propagate_fractures_by_toughness,
    solve_edfm_tedfm_mechanics,
)

__all__ = [
    "FractureLine", "EDFMGrid", "build_edfm", "avg_frac_distance",
    "solve_incomp_edfm", "EDFMMechanicsResult", "build_tedfm_wrapper",
    "fracture_normals", "fracture_normal_strain",
    "solve_edfm_tedfm_mechanics", "FracturePropagationResult",
    "fracture_tip_mode_i_indicator", "propagate_fractures_by_aperture",
    "propagate_fractures_by_toughness", "PlanarFracture", "Projection3D",
    "project_planar_fracture_to_grid", "build_edfm_3d",
    "solve_incomp_edfm_3d",
]
