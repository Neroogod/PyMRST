"""MRST ``Enu2C_AD.m`` elasticity tensor as a sparse AD tensor."""

import numpy as np

from pymrst.modules.vemmech.ad_version import SparseMultiArray, _coef_concat, _coef_take


def enu2c_ad(E, nu, g, ixnames=("i", "j", "c")) -> SparseMultiArray:
    """Eager tensor counterpart of MRST ``Enu2C_AD``."""
    if g.griddim != 3:
        raise ValueError("Enu2C_AD only supports 3D grids in MRST")
    vals = []
    ixs = []
    for c in range(g.cells.num):
        Ec = _coef_take(E, c)
        nuc = _coef_take(nu, c)
        fac = Ec / ((1.0 + nuc) * (1.0 - 2.0 * nuc))
        normal_diag = fac * (1.0 - nuc)
        normal_offdiag = fac * nuc
        shear = fac * ((1.0 - 2.0 * nuc) / 2.0)
        for i in range(3):
            for j in range(3):
                vals.append(normal_diag if i == j else normal_offdiag)
                ixs.append([i + 1, j + 1, c + 1])
        for i in range(3, 6):
            vals.append(shear)
            ixs.append([i + 1, i + 1, c + 1])
    return SparseMultiArray(_coef_concat(vals), np.asarray(ixs, dtype=np.int64),
                            ixnames)

__all__ = ["enu2c_ad"]
