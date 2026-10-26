"""MRST ``makeModel3``: a two-fault synthetic corner-point model (GRDECL).

MRST seeds MATLAB's generator with `rng(0)` and perturbs each layer surface
with `rand` and `randn` draws. `rand` is reproduced exactly here (MATLAB's
Mersenne twister after `rng(0)` is NumPy's `RandomState(5489)`); MATLAB's
`randn` uses its own Ziggurat transform, which NumPy does not provide, so the
normal draws - and with them the surfaces - differ from MRST's unless
MATLAB's own draws are passed as `draws=(a, s)` (a PyMRST addition), with
which the model is MRST's exactly.
"""

import numpy as np

from pymrst.gridprocessing.testgrids._corner_point import base_grdecl, logical_grid


def _matlab_round(x):
    """MATLAB's `round`: halves away from zero."""
    x = np.asarray(x, dtype=float)
    return (np.sign(x)*np.floor(np.abs(x) + 0.5)).astype(int)


def make_model3(dims, phys_dims=(1000.0, 600.0, 15.0), *, draws=None):
    """MRST `g = makeModel3(dims, physDims)`; `draws = (a, s)` are MATLAB's
    `rand(nz+1, 1)` and `randn(nz+1, 2)` after `rng(0)`."""
    dims, (x, y, z) = logical_grid(dims, phys_dims)
    phys_dims = np.asarray(phys_dims, dtype=float)
    if draws is None:
        rng = np.random.RandomState(5489)
        a = rng.random_sample(dims[2] + 1)
        s = rng.standard_normal(2*(dims[2] + 1)).reshape((dims[2] + 1, 2), order="F")
    else:
        a = np.asarray(draws[0], dtype=float).ravel()
        s = np.asarray(draws[1], dtype=float).reshape((dims[2] + 1, 2))
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
    i1, j1 = _matlab_round([.6 * phys_dims[0] / spacing[0],
                            .5 * phys_dims[1] / spacing[1]])
    i2, j2 = _matlab_round([.5 * phys_dims[0] / spacing[0],
                            (1 / 3) * phys_dims[1] / spacing[1]])
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
