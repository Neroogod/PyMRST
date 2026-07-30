"""MRST ``oneSlopingFault`` fixture."""

import numpy as np

from pymrst.gridprocessing.testgrids._corner_point import base_grdecl, logical_grid


def one_sloping_fault(dims, drop=0.0):
    dims, (x, y, z) = logical_grid(dims, (1.0, 1.0, 1.0))
    if dims[0] % 2:
        raise ValueError("number of x cells must be even")
    x = (x + 0.2 * (0.5 - np.abs(x - 0.5)) * (z - 0.5)) * 500.0
    y, z = y * 100.0, z * 32.0
    out = base_grdecl(dims, x, y, z)
    zz = out["ZCORN"].reshape(tuple(2 * dims), order="F")
    zz[dims[0]:, :, :] += float(drop)
    out["ZCORN"] = zz.ravel(order="F")
    return out


oneSlopingFault = one_sloping_fault
