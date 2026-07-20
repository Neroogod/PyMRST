"""Poroelastic constants (MRST ``poroParams.m`` layout)."""

from typing import Dict


def poro_params(E: float, nu: float, alpha: float, M: float) -> Dict[str, float]:
    """Return drained/undrained poroelastic constants."""
    K = E / (3.0 * (1.0 - 2.0 * nu))
    G = E / (2.0 * (1.0 + nu))
    K_u = K + alpha**2 * M
    B = alpha * M / K_u
    nu_u = (3.0 * K_u - 2.0 * G) / (2.0 * (3.0 * K_u + G))
    return {"K": K, "G": G, "K_u": K_u, "B": B, "nu_u": nu_u}


def consolidation_coefficient(
    perm: float,
    mu_f: float,
    E: float,
    nu: float,
    alpha: float,
    M: float,
) -> float:
    """Mandel consolidation coefficient."""
    p = poro_params(E, nu, alpha, M)
    G, B, nu_u = p["G"], p["B"], p["nu_u"]
    return (2.0 * perm * B**2 * G * (1.0 - nu) * (1.0 + nu_u) ** 2) / (
        9.0 * mu_f * (1.0 - nu_u) * (nu_u - nu)
    )


__all__ = ["consolidation_coefficient", "poro_params"]
