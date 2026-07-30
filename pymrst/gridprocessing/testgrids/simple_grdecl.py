"""MRST ``simpleGrdecl`` faulted corner-point fixture."""

import numpy as np

from pymrst.gridprocessing.testgrids._corner_point import (
    base_grdecl,
    logical_grid,
)


def simple_grdecl(
    dims,
    drop=None,
    *,
    flat=False,
    undisturbed=False,
    phys_dims=(1.0, 1.0, 0.5),
):
    dims, (x, y, z) = logical_grid(dims, phys_dims)
    phys_dims = np.asarray(phys_dims, dtype=float)
    if not undisturbed:
        x = x + 0.2 * (0.5 - np.abs(x - 0.5)) * (z - 0.5)
    if not flat and not undisturbed:
        for k in range(dims[2] + 1):
            xi, eta = x[:, :, k] / phys_dims[0], y[:, :, k] / phys_dims[1]
            z[:, :, k] += (
                -0.05 * np.sin(np.pi * (xi - 0.5))
                - 0.075 * np.sin(np.pi * (eta + 2 * xi))
            )
        z.sort(axis=2)
    out = base_grdecl(dims, x, y, z)
    if drop is not None:
        zz = out["ZCORN"].reshape(tuple(2 * dims), order="F")
        right = slice(dims[0], None)
        if callable(drop):
            values = np.asarray(drop(np.linspace(0, phys_dims[1], dims[1] + 1)))
            zz[right, 0::2, :] += values[None, :, None]
            if dims[1] > 1:
                zz[right, 1:-1:2, :] = zz[right, 2::2, :]
        else:
            zz[right, :, :] += float(drop)
        out["ZCORN"] = zz.ravel(order="F")
    return out


simpleGrdecl = simple_grdecl
