"""van Genuchten-Mualem theta/psi curves (MRST ``vGM_theta.m``)."""

import numpy as np

from pymrst.modules.fv_unsat.utility._parameters import get_parameter


def vgm_theta(phys):
    """Return ``theta(psi)``, ``krw(psi)``, and ``C_theta(psi)`` callables."""
    alpha = float(get_parameter(phys, "alpha"))
    theta_r = float(get_parameter(phys, "theta_r"))
    theta_s = float(get_parameter(phys, "theta_s"))
    n = float(get_parameter(phys, "n"))
    try:
        m = float(get_parameter(phys, "m"))
    except (KeyError, AttributeError):
        m = 1.0 - 1.0/n

    def theta(psi):
        p = np.asarray(psi, dtype=float)
        unsat = p < 0
        effective = (1.0 + (alpha*np.abs(p))**n)**(-m)
        return np.where(unsat, (theta_s - theta_r)*effective + theta_r, theta_s)

    def capacity(psi):
        p = np.asarray(psi, dtype=float)
        a = np.abs(p)
        with np.errstate(divide="ignore", invalid="ignore"):
            out = (
                m*n*p*(theta_r - theta_s)*alpha**n*a**(n - 2.0)
                / (alpha**n*a**n + 1.0)**(m + 1.0)
            )
        return np.where(p < 0, np.nan_to_num(out), 0.0)

    def relative_permeability(psi):
        p = np.asarray(psi, dtype=float)
        x = alpha*np.abs(p)
        out = (
            (1.0 - x**(n - 1.0)*(1.0 + x**n)**(-m))**2
            / (1.0 + x**n)**(m/2.0)
        )
        return np.where(p < 0, out, 1.0)

    return theta, relative_permeability, capacity


vGM_theta = vgm_theta

__all__ = ["vgm_theta", "vGM_theta"]
