"""MRST ``calculate_stress.m`` calibration stress helper."""

from __future__ import annotations

import numpy as np

from pymrst.modules.vemmech.vem import calculate_stress_vem, enu2c


def _dirichlet_values_from_bc(g, bc):
    dim = g.griddim
    names = bc.disp_bc.index_names()
    d_col = names.index("d")
    n_col = names.index("n")
    ixs = np.asarray(bc.disp_bc.ixs, dtype=np.int64)
    vals = np.asarray(bc.disp_bc.coefs, dtype=float)
    dofs = (ixs[:, d_col] - 1) + dim * (ixs[:, n_col] - 1)
    keep = ~np.isnan(vals)
    return dofs[keep], vals[keep]


def calculate_stress(u, dd, g, bcfun, efun, nufun, op=None):
    """Return calibration stress values for controls and displacement."""
    dim = g.griddim
    uu = np.asarray(dd, dtype=float)
    if uu.ndim == 2:
        full = uu[:, :dim].reshape(-1).copy()
    else:
        full = uu.ravel().copy()
    if full.size != dim * g.nodes.num:
        raise ValueError("dd must contain one displacement vector per node")

    bc = bcfun(u)
    dirdofs, dirvals = _dirichlet_values_from_bc(g, bc)
    full[dirdofs] = dirvals

    if op is not None and isinstance(op, dict) and "stress_op" in op:
        stress_op = op["stress_op"]
        C = None
    else:
        stress_op = None
        C = enu2c(efun(u), nufun(u))
    stress, _ = calculate_stress_vem(g, full, C=C, stress_op=stress_op)
    return stress.reshape(-1)


__all__ = ["calculate_stress"]
