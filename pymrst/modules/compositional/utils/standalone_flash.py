"""Standalone compositional flash (MRST ``standaloneFlash.m`` layout)."""

from dataclasses import dataclass

import numpy as np

from pymrst.modules.compositional.eos import Mixture, fugacity_coefficients
from pymrst.modules.compositional.utils.estimate_equilibrium_wilson import wilson_k
from pymrst.modules.compositional.utils.solve_rachford_rice_vle import rachford_rice


@dataclass
class FlashResult:
    """Two-phase equilibrium at pressure, temperature, and overall composition."""

    V: np.ndarray
    x: np.ndarray
    y: np.ndarray
    K: np.ndarray
    Z_L: np.ndarray
    Z_V: np.ndarray
    two_phase: np.ndarray
    iterations: int


def flash(
    mix: Mixture,
    p: np.ndarray,
    T: np.ndarray,
    z: np.ndarray,
    tol: float = 1e-10,
    max_it: int = 200,
) -> FlashResult:
    """Two-phase Peng-Robinson flash by successive substitution."""
    p = np.atleast_1d(np.asarray(p, float))
    T = np.atleast_1d(np.asarray(T, float))
    z = np.atleast_2d(np.asarray(z, float))
    p, T = np.broadcast_arrays(p, T)
    if z.shape[0] == 1 and p.size > 1:
        z = np.repeat(z, p.size, axis=0)

    K = wilson_k(mix, p, T)
    single_liquid = np.sum(z * K, axis=1) <= 1.0
    single_vapour = np.sum(z / K, axis=1) <= 1.0
    active = ~(single_liquid | single_vapour)

    V = rachford_rice(z, K)
    V[single_liquid] = 0.0
    V[single_vapour] = 1.0

    it = 0
    for it in range(1, max_it + 1):
        x = z / (1.0 + V[:, None] * (K - 1.0))
        y = K * x
        x = x / x.sum(axis=1, keepdims=True)
        y = y / y.sum(axis=1, keepdims=True)

        ln_phi_l, Z_L = fugacity_coefficients(mix, p, T, x, vapour=False)
        ln_phi_v, Z_V = fugacity_coefficients(mix, p, T, y, vapour=True)

        K_new = np.exp(ln_phi_l - ln_phi_v)
        err = float(np.abs(np.log(K_new[active] / K[active])).max() if active.any() else 0.0)
        K[active] = K_new[active]

        trivial = np.sum(np.log(K) ** 2, axis=1) < 1e-8
        newly_single = active & trivial
        if newly_single.any():
            liq = np.sum(z * K, axis=1) <= 1.0
            V[newly_single & liq] = 0.0
            V[newly_single & ~liq] = 1.0
            active = active & ~newly_single

        Vn = rachford_rice(z, K)
        V[active] = Vn[active]
        if err < tol or not active.any():
            break

    x = z / (1.0 + V[:, None] * (K - 1.0))
    y = K * x
    x = x / x.sum(axis=1, keepdims=True)
    y = y / y.sum(axis=1, keepdims=True)
    _, Z_L = fugacity_coefficients(mix, p, T, x, vapour=False)
    _, Z_V = fugacity_coefficients(mix, p, T, y, vapour=True)
    two_phase = (V > 1e-8) & (V < 1.0 - 1e-8)

    return FlashResult(
        V=V,
        x=x,
        y=y,
        K=K,
        Z_L=Z_L,
        Z_V=Z_V,
        two_phase=two_phase,
        iterations=it,
    )


def standaloneFlash(p, T, z, model, tol: float = 1e-10, max_it: int = 200) -> FlashResult:
    """MRST camelCase entry point for standalone compositional flash."""
    mix = getattr(model, "CompositionalMixture", model)
    mix = getattr(model, "mixture", mix)
    return flash(mix, p, T, z, tol=tol, max_it=max_it)


def standalone_flash(p, T, z, model, tol: float = 1e-10, max_it: int = 200) -> FlashResult:
    """Pythonic alias for :func:`standaloneFlash`."""
    return standaloneFlash(p, T, z, model, tol=tol, max_it=max_it)


__all__ = ["flash", "FlashResult", "standaloneFlash", "standalone_flash"]
