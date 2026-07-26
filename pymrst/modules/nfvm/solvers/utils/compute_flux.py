"""NFVM face flux reconstruction (MRST ``computeFlux.m``)."""

import numpy as np


def compute_flux(g, pressure, transmissibility):
    """Compute globally oriented face fluxes from NTPFA coefficients."""
    pressure = np.asarray(pressure, dtype=float).ravel()
    T1, T2 = transmissibility
    flux = np.zeros(g.faces.num)
    internal = np.all(g.faces.neighbors >= 0, axis=1)
    c1, c2 = g.faces.neighbors[internal].T
    flux[internal] = T1[internal]*pressure[c1] - T2[internal]*pressure[c2]
    return flux


computeFlux = compute_flux

__all__ = ["compute_flux", "computeFlux"]
