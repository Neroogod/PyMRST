"""MRST ``assembleMPSA.m`` MPSA-W assembly entry point."""

import numpy as np

from pymrst.modules.mpsaw import MPSAAssembly, mpsaw_assemble


def assemble_mpsa(
    g,
    young: float | np.ndarray,
    poisson: float | np.ndarray,
    *,
    eta: float = 0.0,
    body_force: np.ndarray | None = None,
) -> MPSAAssembly:
    """Assemble the cell-centered MPSA-W-style mechanics operator."""
    return mpsaw_assemble(
        g,
        young,
        poisson,
        eta=eta,
        body_force=body_force,
    )

__all__ = ["MPSAAssembly", "assemble_mpsa", "mpsaw_assemble"]
