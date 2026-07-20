"""Pure-water density correlation (MRST ``density_pure_water.m`` layout)."""

from pymrst.modules.geothermal.water_props import (
    Num,
    P_MAX,
    P_MIN,
    P_REF,
    T_MAX_C,
    _clamp,
    _coefficients_pure_water,
    _isothermal_compressibility,
)
from pymrst.utils.adi import ad_exp


def density_pure_water(p: Num, t: Num) -> Num:
    """Pure-water density in kg/m3 using MRST's Spivey implementation."""
    p = _clamp(p, P_MIN, P_MAX)
    t_c = _clamp(t - 273.15, 1e-3, T_MAX_C)

    rho_w0, ew, fw = _coefficients_pure_water(t_c)
    iw = _isothermal_compressibility(p, P_REF, ew, fw)
    iw0 = _isothermal_compressibility(P_REF, P_REF, ew, fw)
    return rho_w0 * ad_exp(iw - iw0) * 1000.0


__all__ = ["density_pure_water"]
