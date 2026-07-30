"""Corner-point pillar coordinates (MRST ``buildCornerPtPillars``)."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def build_corner_pt_pillars(
    grdecl: Mapping,
    *,
    coincidence_tolerance: float = 100 * np.finfo(float).eps,
    scale: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return pillar endpoint arrays ``(X, Y, Z)``, each shaped ``(np, 2)``.

    ``COORD`` contains ``[x_top, y_top, z_top, x_bottom, y_bottom,
    z_bottom]`` for every pillar. Collapsed pillars are extended over the
    global ZCORN depth interval. With ``scale=True``, all other pillars are
    linearly extended to that same interval.
    """
    coord = np.asarray(grdecl["COORD"], dtype=float)
    if coord.size % 6:
        raise ValueError("COORD must contain six values per pillar")
    lines = coord.reshape(-1, 6)
    zcorn = np.asarray(grdecl["ZCORN"], dtype=float).ravel()
    if zcorn.size == 0:
        raise ValueError("ZCORN cannot be empty")

    x = lines[:, [0, 3]].copy()
    y = lines[:, [1, 4]].copy()
    z = lines[:, [2, 5]].copy()
    z_min, z_max = float(zcorn.min()), float(zcorn.max())

    collapsed = np.abs(z[:, 1] - z[:, 0]) < abs(coincidence_tolerance)
    z[collapsed] = np.array([z_min, z_max])

    if scale:
        dz = z[:, 1] - z[:, 0]
        if np.any(np.abs(dz) < abs(coincidence_tolerance)):
            raise ValueError("cannot scale a pillar with zero depth extent")
        # Preserve MRST 2024a's observable sequential update exactly: X[:, 0]
        # is changed before X[:, 1] is evaluated, and only Y[:, 0] is scaled.
        # This is slightly asymmetric but is pinned by the MATLAB reference.
        x[:, 0] += (
            (z_min - z[:, 0]) * (x[:, 1] - x[:, 0]) / dz
        )
        x[:, 1] = (
            x[:, 0]
            + (z_max - z[:, 0]) * (x[:, 1] - x[:, 0]) / dz
        )
        y[:, 0] += (
            (z_min - z[:, 0]) * (y[:, 1] - y[:, 0]) / dz
        )
        z = np.tile([z_min, z_max], (z.shape[0], 1))

    return x, y, z


buildCornerPtPillars = build_corner_pt_pillars

__all__ = ["build_corner_pt_pillars", "buildCornerPtPillars"]
