"""Upstream hydraulic-head weighting from ``upstreamMPFA_hydHead.m``."""

import numpy as np


def upstream_mpfa_hyd_head(
    krw, F, boundF, g_struct, bc_struct, bc_val,
    hyd_head, zeta_cells, zeta_faces,
):
    """Select face relative permeability from the upstream pressure head."""
    head = np.asarray(hyd_head, dtype=float).ravel()
    zc = np.asarray(zeta_cells, dtype=float).ravel()
    zf = np.asarray(zeta_faces, dtype=float).ravel()
    neighbors = np.asarray(g_struct.faces.neighbors, dtype=np.int64)
    internal = np.all(neighbors >= 0, axis=1)
    flux = np.asarray(F(head) + boundF(bc_val), dtype=float).ravel()
    out = np.zeros(g_struct.faces.num)

    for face, kind, value in zip(
        bc_struct.face, bc_struct.type, bc_struct.value
    ):
        face = int(face)
        if kind == "flux":
            out[face] = 1.0
            continue
        if kind != "pressure":
            raise ValueError(f"unsupported boundary type {kind!r}")
        n0, n1 = map(int, neighbors[face])
        if flux[face] >= 0:
            use_boundary = n0 < 0
            cell = n0 if n0 >= 0 else n1
        else:
            use_boundary = n1 < 0
            cell = n1 if n1 >= 0 else n0
        pressure_head = (
            float(value) - zf[face]
            if use_boundary else head[cell] - zc[cell]
        )
        out[face] = float(np.asarray(krw(pressure_head)))

    faces = np.flatnonzero(internal)
    n0 = neighbors[faces, 0]
    n1 = neighbors[faces, 1]
    upstream = np.where(flux[faces] >= 0, n0, n1)
    out[faces] = np.asarray(krw(head[upstream] - zc[upstream]))
    return out


upstreamMPFA_hydHead = upstream_mpfa_hyd_head

__all__ = ["upstream_mpfa_hyd_head", "upstreamMPFA_hydHead"]
