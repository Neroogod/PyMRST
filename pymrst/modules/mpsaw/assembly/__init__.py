"""MRST-shaped `mpsaw/assembly` namespace."""

from pymrst.modules.mpsaw import (
    MPSAAssembly,
    MPSAInteractionRegion,
    mpsaw_assemble,
    mpsaw_divergence_operator,
    mpsaw_interaction_regions,
)
from pymrst.modules.mpsaw.assembly.core_mpsa_assembly import core_mpsa_assembly

assemble_mpsa = mpsaw_assemble

__all__ = [
    "MPSAAssembly",
    "MPSAInteractionRegion",
    "assemble_mpsa",
    "core_mpsa_assembly",
    "mpsaw_assemble",
    "mpsaw_divergence_operator",
    "mpsaw_interaction_regions",
]
