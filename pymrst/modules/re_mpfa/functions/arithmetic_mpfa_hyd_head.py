"""Arithmetic hydraulic-head weighting from ``arithmeticMPFA_hydHead.m``."""

import numpy as np


def arithmetic_mpfa_hyd_head(
    krw, g_struct, bc_struct, hyd_head, zeta_cells, zeta_faces
):
    """Compute face relative permeability from hydraulic head.

    Hydraulic head is converted to pressure head at every evaluation point,
    exactly as in MRST's re-mpfa implementation.
    """
    head = np.asarray(hyd_head, dtype=float).ravel()
    zc = np.asarray(zeta_cells, dtype=float).ravel()
    zf = np.asarray(zeta_faces, dtype=float).ravel()
    neighbors = np.asarray(g_struct.faces.neighbors, dtype=np.int64)
    internal = np.all(neighbors >= 0, axis=1)
    out = np.zeros(g_struct.faces.num)

    for face, kind, value in zip(
        bc_struct.face, bc_struct.type, bc_struct.value
    ):
        face = int(face)
        if kind == "flux":
            out[face] = 1.0
        elif kind == "pressure":
            cell = int(neighbors[face, neighbors[face] >= 0][0])
            out[face] = 0.5*(
                float(np.asarray(krw(float(value) - zf[face])))
                + float(np.asarray(krw(head[cell] - zc[cell])))
            )
        else:
            raise ValueError(f"unsupported boundary type {kind!r}")

    c1, c2 = neighbors[internal].T
    out[internal] = 0.5*(
        np.asarray(krw(head[c1] - zc[c1]))
        + np.asarray(krw(head[c2] - zc[c2]))
    )
    return out


arithmeticMPFA_hydHead = arithmetic_mpfa_hyd_head

__all__ = ["arithmetic_mpfa_hyd_head", "arithmeticMPFA_hydHead"]
