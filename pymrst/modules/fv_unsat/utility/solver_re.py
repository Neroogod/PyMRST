"""Newton solver for Richards' equation (MRST ``solverRE.m``)."""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import root


@dataclass
class RichardsStep:
    pressure_head: np.ndarray
    previous_iteration: np.ndarray
    iterations: int
    residual_norm: float
    face_flux: np.ndarray


def solver_re(
    psi_n,
    model_eqs,
    dt,
    source=None,
    *,
    tolerance=1e-9,
    max_iterations=25,
):
    """Solve one fully implicit Richards time step."""
    psi_n = np.asarray(psi_n, dtype=float).ravel()
    previous = psi_n.copy()
    result = root(
        lambda psi: model_eqs.residual(psi, psi_n, dt, source),
        previous,
        method="hybr",
        options={"maxfev": max_iterations*(psi_n.size + 1), "xtol": tolerance},
    )
    residual_norm = float(np.linalg.norm(
        model_eqs.residual(result.x, psi_n, dt, source)
    ))
    if not result.success and residual_norm > tolerance:
        raise RuntimeError(f"Richards solver failed: {result.message}")
    face_flux, _ = model_eqs.flux(result.x)
    return RichardsStep(
        result.x,
        previous,
        int(getattr(result, "nfev", 0)),
        residual_norm,
        face_flux,
    )


solverRE = solver_re

__all__ = ["RichardsStep", "solver_re", "solverRE"]
