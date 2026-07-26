"""Pressure-dependent NTPFA transmissibilities (MRST ``TransNTPFA.m``)."""

import numpy as np
import scipy.sparse as sp

from pymrst.utils.adi import ad_matmul, value


def trans_ntpfa(g, pressure, os_flux):
    """Return the two nonlinear face transmissibility vectors."""
    is_ad = hasattr(pressure, "jac")
    pressure_value = np.asarray(value(pressure), dtype=float).ravel()
    internal = np.flatnonzero(np.all(g.faces.neighbors >= 0, axis=1))
    tii = [np.zeros((g.faces.num, 2)), np.zeros((g.faces.num, 2))]
    correction_maps = []
    correction_constants = []
    for side in range(2):
        rows, cols, coefficients = [], [], []
        constant = np.zeros(g.faces.num)
        for face in internal:
            stencil = os_flux[face, side]
            tii[side][face] = stencil[:2, 1]
            for cell, coefficient in stencil[2:]:
                cell = int(cell)
                if cell >= 0:
                    rows.append(int(face))
                    cols.append(cell)
                    coefficients.append(float(coefficient))
                else:
                    constant[face] += float(coefficient)
        correction_maps.append(sp.csr_array(
            (coefficients, (rows, cols)), shape=(g.faces.num, g.cells.num)
        ))
        correction_constants.append(constant)

    correction = [
        ad_matmul(correction_maps[side], pressure) + correction_constants[side]
        if is_ad else correction_maps[side] @ pressure_value + correction_constants[side]
        for side in range(2)
    ]

    scale = max(
        max((np.max(np.abs(m.data)) if m.nnz else 0.0)
            for m in correction_maps),
        1.0,
    )
    tolerance = 1e-12*scale
    nonzero = [
        (np.abs(np.asarray(value(r)).ravel()) > tolerance).astype(float)
        for r in correction
    ]
    correction = [correction[i]*nonzero[i] for i in range(2)]
    denom = correction[0] + correction[1]
    active = (
        np.abs(np.asarray(value(denom)).ravel()) > tolerance
    ).astype(float)
    safe_denom = denom + (1.0 - active)
    mu1 = active*(correction[1]/safe_denom) + (1.0 - active)*0.5
    mu2 = 1.0 - mu1
    mu1_value = np.asarray(value(mu1)).ravel()
    if np.any((mu1_value < -1e-12) | (mu1_value > 1 + 1e-12)):
        raise ValueError("NTPFA nonlinear convex weight left [0, 1]")

    T1 = (
        mu1*tii[0][:, 0]
        + mu2*tii[1][:, 1]
    )
    T2 = (
        mu1*tii[0][:, 1]
        + mu2*tii[1][:, 0]
    )
    return T1, T2


TransNTPFA = trans_ntpfa

__all__ = ["trans_ntpfa", "TransNTPFA"]
