"""Brine density correlation (MRST ``density_brine.m`` layout)."""

from pymrst.modules.geothermal.water_props import (
    Num,
    P_MAX,
    P_MIN,
    P_REF,
    T_MAX_C,
    T_MIN_C,
    _clamp,
    _coefficients_brine,
    _coefficients_pure_water,
    _density_brine_ref,
    _isothermal_compressibility,
    _molality,
)
from pymrst.utils.adi import ad_exp


def density_brine(p: Num, t: Num, c: Num = 0.0) -> Num:
    """Brine density in kg/m3 using MRST's Spivey implementation."""
    p = _clamp(p, P_MIN, P_MAX)
    t_c = _clamp(t - 273.15, T_MIN_C, T_MAX_C)
    mol = _molality(c)

    rho_w0, ew, fw = _coefficients_pure_water(t_c)
    rho_b0 = _density_brine_ref(rho_w0, t_c, mol)
    eb, fb = _coefficients_brine(ew, fw, mol, t_c)

    ib = _isothermal_compressibility(p, P_REF, eb, fb)
    ib0 = _isothermal_compressibility(P_REF, P_REF, eb, fb)
    return rho_b0 * ad_exp(ib - ib0) * 1000.0


__all__ = ["density_brine"]
