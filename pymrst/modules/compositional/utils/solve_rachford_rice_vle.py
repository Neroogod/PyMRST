"""Rachford-Rice VLE solve (MRST ``solveRachfordRiceVLE.m`` layout)."""

import numpy as np


def rachford_rice(
    z: np.ndarray,
    K: np.ndarray,
    tol: float = 1e-12,
    max_it: int = 100,
) -> np.ndarray:
    """Vapour fraction solving the Rachford-Rice equation.

    The canonical implementation lives in :mod:`compositional.eos`; this
    MRST-layout utility delegates to it so every flash path uses the same
    vectorized active-row solver.
    """
    from pymrst.modules.compositional.eos import rachford_rice as _active_solver

    return _active_solver(z, K, tol=tol, max_it=max_it)


def solveRachfordRiceVLE(z: np.ndarray, K: np.ndarray, tol: float = 1e-12, max_it: int = 100) -> np.ndarray:
    """MRST camelCase entry point for the Rachford-Rice VLE solve."""
    return rachford_rice(z, K, tol=tol, max_it=max_it)


def solve_rachford_rice_vle(z: np.ndarray, K: np.ndarray, tol: float = 1e-12, max_it: int = 100) -> np.ndarray:
    """Pythonic alias for :func:`solveRachfordRiceVLE`."""
    return solveRachfordRiceVLE(z, K, tol=tol, max_it=max_it)


__all__ = ["rachford_rice", "solveRachfordRiceVLE", "solve_rachford_rice_vle"]
