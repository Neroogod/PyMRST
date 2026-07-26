"""van Genuchten-Mualem saturation/pressure curves (MRST ``vGM_saturation.m``)."""

import numpy as np

from pymrst.modules.fv_unsat.utility._parameters import get_parameter


def vgm_saturation(phys):
    """Return ``Sw(p)``, ``krw(p)``, and specific capacity ``C_S(p)``."""
    alpha = float(get_parameter(phys, "alpha"))
    rho = float(get_parameter(phys, "rho"))
    gravity = float(get_parameter(phys, "g"))
    poro = float(get_parameter(phys, "poro"))
    theta_r = float(get_parameter(phys, "theta_r"))
    n = float(get_parameter(phys, "n"))
    try:
        m = float(get_parameter(phys, "m"))
    except (KeyError, AttributeError):
        m = 1.0 - 1.0/n
    acoef = alpha/(rho*gravity)
    residual = theta_r/poro

    def saturation(pressure):
        p = np.asarray(pressure, dtype=float)
        effective = (1.0 + (acoef*np.abs(p))**n)**(-m)
        return np.where(p < 0, (1.0 - residual)*effective + residual, 1.0)

    def capacity(pressure):
        p = np.asarray(pressure, dtype=float)
        a = np.abs(p)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (
                -m*n*(1.0 - residual)*p*(acoef*a)**n
                / (a**2*(1.0 + (acoef*a)**n)**(m + 1.0))
            )
        return np.where(p < 0, np.nan_to_num(out), 0.0)

    def relative_permeability(pressure):
        p = np.asarray(pressure, dtype=float)
        x = acoef*np.abs(p)
        out = (
            (1.0 - x**(n - 1.0)*(1.0 + x**n)**(-m))**2
            / (1.0 + x**n)**(m/2.0)
        )
        return np.where(p < 0, out, 1.0)

    return saturation, relative_permeability, capacity


vGM_saturation = vgm_saturation

__all__ = ["vgm_saturation", "vGM_saturation"]
