"""MRST ``makeModel3`` two-fault synthetic GRDECL."""

import numpy as np

from pymrst.gridprocessing.testgrids._corner_point import base_grdecl, logical_grid


def make_model3(dims, phys_dims=(1000.0, 600.0, 15.0), *, seed=0):
    dims, (x, y, z) = logical_grid(dims, phys_dims)
    phys_dims = np.asarray(phys_dims, dtype=float)
    rng = np.random.default_rng(seed)
    a = rng.random(dims[2] + 1)
    s = rng.standard_normal((dims[2] + 1, 2))
    for k in range(dims[2] + 1):
        xi, eta = x[:, :, k] / phys_dims[0], y[:, :, k] / phys_dims[1]
        fun = (
            -5 * np.sin(np.pi * (xi - .5))
            - 3 * np.sin(np.pi * (eta + 2 * xi))
            + a[k] / 2 * np.sin(2 * np.pi * xi + s[k, 0])
            * np.sin(4 * np.pi * eta + s[k, 1])
        )
        z[:, :, k] -= fun
    out = base_grdecl(dims, x, y, z)
    zz = out["ZCORN"].reshape(tuple(2 * dims), order="F")
    spacing = phys_dims / dims
    i1, j1 = np.rint([.6 * phys_dims[0] / spacing[0],
                      .5 * phys_dims[1] / spacing[1]]).astype(int)
    i2, j2 = np.rint([.5 * phys_dims[0] / spacing[0],
                      (1 / 3) * phys_dims[1] / spacing[1]]).astype(int)
    zz[:2*i1, :2*j1, :] -= 2
    zz[2*i2:, 2*j2:, :] += 3
    zz = np.cumsum(np.concatenate([
        zz[:, :, :1], np.maximum(0, np.diff(zz, axis=2))
    ], axis=2), axis=2)
    out["ZCORN"] = zz.ravel(order="F")
    cx, cy, _ = np.meshgrid(
        np.linspace(0, 1, dims[0]),
        np.linspace(0, 1, dims[1]),
        np.linspace(0, 1, dims[2]),
        indexing="ij",
    )
    active = np.ones(tuple(dims), dtype=np.int32)
    active[(cx - .4) ** 2 + cy ** 2 > .95] = 0
    active[(cx - .6) ** 2 + (cy - 1) ** 2 > .9] = 0
    out["ACTNUM"] = active.ravel(order="F")
    return out


makeModel3 = make_model3
