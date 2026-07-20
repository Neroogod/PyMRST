"""Backend-neutral helper corresponding to MRST ``surfWithOutline.m``."""

from __future__ import annotations

import numpy as np


def surf_with_outline(x, y, z):
    """Return surface arrays and outline polylines for plotting elsewhere."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    if x.shape != y.shape or x.shape != z.shape:
        raise ValueError("x, y, and z must have identical shapes")
    outline = {
        "bottom": (x[0, :], y[0, :], z[0, :]),
        "top": (x[-1, :], y[-1, :], z[-1, :]),
        "left": (x[:, 0], y[:, 0], z[:, 0]),
        "right": (x[:, -1], y[:, -1], z[:, -1]),
    }
    return {"surface": (x, y, z), "outline": outline}


__all__ = ["surf_with_outline"]
