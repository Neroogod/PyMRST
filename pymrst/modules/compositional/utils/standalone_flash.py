"""Standalone compositional flash (MRST ``standaloneFlash.m`` layout)."""

from dataclasses import dataclass

import numpy as np

from pymrst.modules.compositional.eos import Mixture


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
    """Run the active MRST-style stability/flash implementation."""
    from pymrst.modules.compositional.eos import flash as active_flash

    result = active_flash(mix, p, T, z, tol=tol, max_it=max_it)
    return FlashResult(**result.__dict__)


def standaloneFlash(p, T, z, model, tol: float = 1e-10, max_it: int = 200) -> FlashResult:
    """MRST camelCase entry point for standalone compositional flash."""
    mix = getattr(model, "CompositionalMixture", model)
    mix = getattr(model, "mixture", mix)
    return flash(mix, p, T, z, tol=tol, max_it=max_it)


def standalone_flash(p, T, z, model, tol: float = 1e-10, max_it: int = 200) -> FlashResult:
    """Pythonic alias for :func:`standaloneFlash`."""
    return standaloneFlash(p, T, z, model, tol=tol, max_it=max_it)


__all__ = ["flash", "FlashResult", "standaloneFlash", "standalone_flash"]
