"""Shared construction utilities for MRST-compatible test GRDECLs."""

import numpy as np


def logical_grid(dims, physical_dims):
    dims = np.asarray(dims, dtype=np.int64)
    physical_dims = np.asarray(physical_dims, dtype=float)
    axes = [
        np.linspace(0.0, physical_dims[d], dims[d] + 1) for d in range(3)
    ]
    return dims, np.meshgrid(*axes, indexing="ij")


def corner_indices(n):
    return np.minimum(np.arange(1, 2 * n + 1) // 2, n)


def coord_from_xyz(x, y, z):
    shape = x.shape
    n = shape[0] * shape[1]
    lines = np.zeros((n, 6), dtype=float)
    lines[:, [0, 3]] = x[:, :, [0, -1]].reshape((n, 2), order="F")
    lines[:, [1, 4]] = y[:, :, [0, -1]].reshape((n, 2), order="F")
    lines[:, [2, 5]] = z[:, :, [0, -1]].reshape((n, 2), order="F")
    return lines.ravel()


def zcorn_from_xyz(z, dims):
    ix = [corner_indices(int(n)) for n in dims]
    return z[np.ix_(*ix)].ravel(order="F")


def base_grdecl(dims, x, y, z, *, active=True):
    out = {
        "cartDims": np.asarray(dims, dtype=np.int64),
        "COORD": coord_from_xyz(x, y, z),
        "ZCORN": zcorn_from_xyz(z, dims),
    }
    if active:
        out["ACTNUM"] = np.ones(int(np.prod(dims)), dtype=np.int32)
    return out
