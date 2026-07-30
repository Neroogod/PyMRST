"""MRST ``pinchedLayersGrdecl`` random pinched-layer fixture."""

import numpy as np

from pymrst.gridprocessing.testgrids._corner_point import base_grdecl, logical_grid


def pinched_layers_grdecl(dims, drop=None, *, seed=None):
    """Build layers with locally variable nonnegative thickness.

    ``seed`` makes the MRST fixture's intentionally random construction
    reproducible for tests.
    """
    dims, (x, y, z) = logical_grid(dims, (1.0, 1.0, 0.0))
    rng = np.random.default_rng(seed)
    z[:, :, 0] = .5 + .02 * np.sin(2*np.pi*x[:, :, 0]) * np.sin(2*np.pi*y[:, :, 0])
    for k in range(1, dims[2] + 1):
        thickness = (
            .2 + .5 * rng.random()
            * np.sin(3 * rng.random() * np.pi * x[:, :, k])
            * np.sin(2 * rng.random() * np.pi * y[:, :, k])
        )
        z[:, :, k] = z[:, :, k-1] + np.maximum(0, thickness * .5 / dims[2])
    out = base_grdecl(dims, x, y, z)
    if drop is not None:
        zz = out["ZCORN"].reshape(tuple(2*dims), order="F")
        right = slice(dims[0], None)
        if callable(drop):
            values = np.asarray(drop(np.linspace(0, 1, dims[1] + 1)))
            zz[right, 0::2, :] += values[None, :, None]
            if dims[1] > 1:
                zz[right, 1:-1:2, :] = zz[right, 2::2, :]
        else:
            zz[right, :, :] += float(drop)
        out["ZCORN"] = zz.ravel(order="F")
    return out


pinchedLayersGrdecl = pinched_layers_grdecl
