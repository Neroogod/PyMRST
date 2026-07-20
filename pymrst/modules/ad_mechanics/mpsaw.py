"""Backward-compatible import path for MPSA-W mechanics utilities."""

from pymrst.modules.mpsaw import (
    MPSAAssembly,
    MPSAInteractionRegion,
    MPSAWResult,
    mpsaw_assemble,
    mpsaw_cell_gradient,
    mpsaw_divergence_operator,
    mpsaw_face_tractions,
    mpsaw_interaction_regions,
    mpsaw_isotropic_stress,
    mpsaw_recover,
    mpsaw_residual,
    mpsaw_strain,
)

__all__ = [
    "MPSAAssembly",
    "MPSAInteractionRegion",
    "MPSAWResult",
    "mpsaw_assemble",
    "mpsaw_cell_gradient",
    "mpsaw_divergence_operator",
    "mpsaw_face_tractions",
    "mpsaw_interaction_regions",
    "mpsaw_isotropic_stress",
    "mpsaw_recover",
    "mpsaw_residual",
    "mpsaw_strain",
]
