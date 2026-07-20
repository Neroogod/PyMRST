"""Brine viscosity correlation (MRST ``viscosity_brine.m`` layout)."""

import numpy as np

from pymrst.modules.geothermal.water_props import (
    Num,
    P_MAX,
    P_MIN,
    T_MAX_C,
    T_MIN_C_VISC,
    _clamp,
    _kestin,
    _molality,
)
from pymrst.utils.adi import ad_exp, ad_log, value


def viscosity_brine(p: Num, t: Num, c: Num = 0.0) -> Num:
    """Brine viscosity in Pa s using MRST's Kestin/high-temperature switch."""
    p = _clamp(p, P_MIN, P_MAX)
    p_mpa = p / 1e6
    t_c = _clamp(t - 273.15, T_MIN_C_VISC, T_MAX_C)
    mol = _molality(c)

    mu_125 = _kestin(p_mpa, 125.0, mol)
    mu_k = _kestin(p_mpa, t_c, mol)

    a1, a2, b1, b2 = 0.0173, 0.068, -1.0531, 0.0273
    log_t = ad_log(t_c / 125.0)
    mu_hot = mu_125 * ad_exp(
        (a2 * (p_mpa / 100.0) + a1) * log_t**2
        + (b2 * (p_mpa / 100.0) + b1) * log_t
    )

    use_kestin = value(t_c) < 125.0
    k = np.asarray(use_kestin, dtype=np.float64)
    return (mu_hot * (1.0 - k) + mu_k * k) * 1e-6


__all__ = ["viscosity_brine"]
