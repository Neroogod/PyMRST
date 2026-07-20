"""Kestin brine viscosity helper (MRST ``kestin_brine_viscosity.m``)."""

from pymrst.modules.geothermal.water_props import Num, _pow10


def kestin_brine_viscosity(p_mpa: Num, t_c: Num, mol: Num) -> Num:
    """Kestin et al. brine viscosity in micro-Pa s, matching MRST."""
    cs = 6.044 + 0.28e-2 * t_c + 0.36e-4 * t_c**2
    c_star = mol / cs
    beta_w = (
        -1.297
        + 0.574e-1 * t_c
        - 0.697e-3 * t_c**2
        + 0.447e-5 * t_c**3
        - 0.105e-7 * t_c**4
    )
    beta_se = 0.545 + 0.28e-2 * t_c - beta_w
    beta_star = 2.5 * c_star - 2.0 * c_star**2 + 0.5 * c_star**3
    beta = beta_se * beta_star + beta_w

    a = 0.3324e-1 * mol + 0.3624e-2 * mol**2 - 0.1879e-3 * mol**3
    b = -0.396e-1 * mol + 0.102e-1 * mol**2 - 0.702e-3 * mol**3

    mu_w0_20 = 1002.0
    d = 20.0 - t_c
    log_ratio = (d / (96.0 + t_c)) * (
        1.2378 - 1.303e-3 * d + 3.06e-6 * d**2 + 2.55e-8 * d**3
    )
    mu_w0 = _pow10(log_ratio) * mu_w0_20
    mu_0 = _pow10(a + b * log_ratio) * mu_w0
    return mu_0 * (1.0 + beta * p_mpa * 1e-3)


__all__ = ["kestin_brine_viscosity"]
