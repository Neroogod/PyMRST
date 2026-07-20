"""Reference-pressure brine density (MRST ``density_brine_refP.m``)."""

from pymrst.modules.geothermal.water_props import Num, _ratio


def density_brine_ref_p(rho_w0: Num, t_c: Num, mol: Num) -> Num:
    """Brine density at MRST's 70 MPa reference pressure."""
    T = t_c / 100.0
    dcm2 = _ratio([-7.925e-5, -1.93e-6, -3.42548e-4, 0.0, 0.0], T)
    dcm32 = _ratio([1.0998e-3, -2.8755e-3, -3.5819e-3, -0.72877, 1.92016], T)
    dcm1 = _ratio([-7.6402e-3, 3.6963e-2, 4.36083e-2, -0.333661, 1.185685], T)
    dcm12 = _ratio([3.746e-5, -3.328e-4, -3.346e-4, 0.0, 0.0], T)

    return (
        rho_w0
        + dcm2 * mol**2
        + dcm32 * mol**1.5
        + dcm1 * mol
        + dcm12 * mol**0.5
    )


__all__ = ["density_brine_ref_p"]
