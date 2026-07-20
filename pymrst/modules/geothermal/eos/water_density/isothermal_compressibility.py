"""Isothermal compressibility helper (MRST ``isothermal_compressibility.m``)."""

import numpy as np

from pymrst.modules.geothermal.water_props import Num
from pymrst.utils.adi import ADI, ad_log


def isothermal_compressibility(p: Num, p0: float, e: Num, f: Num) -> Num:
    """MRST's logarithmic Spivey compressibility integral."""
    a = e * (p / p0) + f
    tol = 1e-16
    if isinstance(a, ADI):
        low = a.val < tol
        if low.any():
            v = np.where(low, tol, a.val)
            mask = (~low).astype(np.float64)
            a = ADI(v, [j.multiply(mask[:, None]) for j in a.jac])
    else:
        a = np.where(a < tol, tol, a)
    return (1.0 / e) * ad_log(a)


__all__ = ["isothermal_compressibility"]
