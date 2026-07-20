"""MRST ``Enu2C.m`` isotropic elasticity tensor construction."""

import numpy as np


def enu2c(E: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Return 3D isotropic elasticity tensors in flattened Voigt form."""
    E = np.asarray(E, dtype=np.float64).ravel()
    nu = np.asarray(nu, dtype=np.float64).ravel()
    nc = E.size
    c = np.zeros((nc, 6, 6))
    for i in range(3):
        for j in range(3):
            c[:, i, j] = nu if i != j else 1.0 - nu
        c[:, 3 + i, 3 + i] = (1.0 - 2.0 * nu) / 2.0
    fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return (c * fac[:, None, None]).reshape(nc, 36)


def enu2c_2d(E: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Return 2D plane-strain elasticity tensors in Voigt order ``xx, yy, xy``."""
    E = np.asarray(E, dtype=np.float64).ravel()
    nu = np.asarray(nu, dtype=np.float64).ravel()
    c = np.zeros((E.size, 3, 3))
    c[:, 0, 0] = 1.0 - nu
    c[:, 0, 1] = nu
    c[:, 1, 0] = nu
    c[:, 1, 1] = 1.0 - nu
    c[:, 2, 2] = (1.0 - 2.0 * nu) / 2.0
    fac = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
    return (c * fac[:, None, None]).reshape(E.size, 9)

__all__ = ["enu2c", "enu2c_2d"]
