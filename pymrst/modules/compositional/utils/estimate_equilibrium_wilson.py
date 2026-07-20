"""Wilson K-value estimate (MRST ``estimateEquilibriumWilson.m`` layout)."""

import numpy as np

from pymrst.modules.compositional.eos import Mixture


def wilson_k(mix: Mixture, p: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Wilson's K-value correlation."""
    mix = getattr(mix, "CompositionalMixture", mix)
    mix = getattr(mix, "mixture", mix)
    p = np.atleast_1d(np.asarray(p, float))
    T = np.atleast_1d(np.asarray(T, float))
    Pr = p[:, None] / mix.Pcrit[None, :]
    Tr = T[:, None] / mix.Tcrit[None, :]
    k = np.exp(5.37 * (1.0 + mix.acentric[None, :]) * (1.0 - 1.0 / Tr)) / Pr
    k[~np.isfinite(k)] = 1e3
    return k


def estimateEquilibriumWilson(model, p: np.ndarray, T: np.ndarray) -> np.ndarray:
    """MRST camelCase entry point for Wilson's K-value correlation."""
    return wilson_k(model, p, T)


def estimate_equilibrium_wilson(model, p: np.ndarray, T: np.ndarray) -> np.ndarray:
    """Pythonic alias for :func:`estimateEquilibriumWilson`."""
    return estimateEquilibriumWilson(model, p, T)


__all__ = ["estimateEquilibriumWilson", "estimate_equilibrium_wilson", "wilson_k"]
