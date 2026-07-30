"""MRST ``cartesianGrdecl`` deterministic test-grid generator."""

import numpy as np

from pymrst.gridprocessing.testgrids._corner_point import (
    coord_from_xyz,
    zcorn_from_xyz,
)


def cartesian_grdecl(x, y, z=None, *, depthz=None):
    """Return a variable-spacing Cartesian corner-point description."""
    if z is None:
        z = np.array([0.0, 1.0])
    x, y, z = (np.asarray(v, dtype=float).ravel() for v in (x, y, z))
    dims = np.array([x.size - 1, y.size - 1, z.size - 1], dtype=np.int64)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    if depthz is None:
        depthz = np.zeros((x.size, y.size))
    depthz = np.asarray(depthz, dtype=float)
    if depthz.shape != (x.size, y.size):
        raise ValueError("depthz must have shape (len(x), len(y))")
    zz = zz + depthz[:, :, None]
    return {
        "cartDims": dims,
        "COORD": coord_from_xyz(xx, yy, zz),
        "ZCORN": zcorn_from_xyz(zz, dims),
    }


cartesianGrdecl = cartesian_grdecl
