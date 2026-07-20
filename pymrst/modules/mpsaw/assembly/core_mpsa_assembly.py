"""MRST ``coreMpsaAssembly.m`` MPSA-W core assembly entry point."""

import numpy as np

from pymrst.modules.mpsaw import (
    MPSAAssembly,
    MPSAInteractionRegion,
    mpsaw_assemble,
    mpsaw_interaction_regions,
)


def core_mpsa_assembly(
    g,
    young: float | np.ndarray,
    poisson: float | np.ndarray,
    *,
    eta: float = 0.0,
    body_force: np.ndarray | None = None,
) -> MPSAAssembly:
    """Assemble MPSA-W operators and interaction-region metadata."""
    return mpsaw_assemble(
        g,
        young,
        poisson,
        eta=eta,
        body_force=body_force,
    )

__all__ = [
    "MPSAAssembly",
    "MPSAInteractionRegion",
    "core_mpsa_assembly",
    "mpsaw_assemble",
    "mpsaw_interaction_regions",
]
