"""Compositional phase properties (MRST ``CompositionalPropertyModel.m``)."""

from typing import Any, Optional

import numpy as np

from pymrst.modules.compositional.eos import OMEGA_B, Mixture
from pymrst.modules.compositional.properties import (
    CENTIPOISE,
    GRAM,
    LBC_COEFFS,
    PSIA,
    RANKINE,
    R_GAS,
    _mix,
)
from pymrst.modules.compositional.fluids.property_model import PropertyModel
from pymrst.utils.adi import value


def molar_density(
    mix: Mixture,
    p,
    T,
    x,
    Z,
    volume_shift: Optional[np.ndarray] = None,
    omega_b: float = OMEGA_B,
) -> Any:
    """Molar density [mol/m3], with optional PR volume shift."""
    v = (R_GAS * T) * Z / p
    if volume_shift is not None:
        beta = omega_b * R_GAS * mix.Tcrit / mix.Pcrit
        v = v - _mix(x, np.asarray(volume_shift, float) * beta)
    return 1.0 / v


def mass_density(
    mix: Mixture,
    p,
    T,
    x,
    Z,
    volume_shift: Optional[np.ndarray] = None,
    omega_b: float = OMEGA_B,
) -> Any:
    """Mass density [kg/m3]."""
    m = _mix(x, mix.molarMass)
    return molar_density(mix, p, T, x, Z, volume_shift, omega_b=omega_b) * m


def viscosity_lbc(
    mix: Mixture,
    p,
    T,
    x,
    Z,
    volume_shift: Optional[np.ndarray] = None,
    omega_b: float = OMEGA_B,
) -> Any:
    """Lohrenz-Bray-Clark viscosity [Pa s]."""
    t_val = np.asarray(value(T), dtype=float)
    m_gram = mix.molarMass / GRAM
    sqrt_m = np.sqrt(m_gram)
    tc_r = mix.Tcrit / RANKINE
    pc_psi = mix.Pcrit / PSIA

    xi = (
        5.4402
        * tc_r ** (1.0 / 6.0)
        / (sqrt_m * pc_psi ** (2.0 / 3.0) * CENTIPOISE)
    )

    num = None
    den = None
    for i in range(mix.n):
        tr = t_val / mix.Tcrit[i]
        mu_i = np.where(
            tr > 1.5,
            17.78e-5 * np.maximum(4.58 * tr - 1.67, 0.0) ** 0.625,
            34e-5 * tr ** 0.94,
        ) / xi[i]
        xi_ = x[i] if isinstance(x, (list, tuple)) else np.asarray(x, float)[:, i]
        term_n = xi_ * (mu_i * sqrt_m[i])
        term_d = xi_ * sqrt_m[i]
        num = term_n if num is None else num + term_n
        den = term_d if den is None else den + term_d
    mu_atm = num / den

    t_pc = _mix(x, mix.Tcrit)
    p_pc = _mix(x, mix.Pcrit)
    v_c = _mix(x, mix.Vcrit)
    m_mix = _mix(x, mix.molarMass)

    xi_mix = (
        5.4402
        * (t_pc / RANKINE) ** (1.0 / 6.0)
        / ((m_mix / GRAM) ** 0.5 * (p_pc / PSIA) ** (2.0 / 3.0) * CENTIPOISE)
    )

    rho_r = v_c * molar_density(mix, p, T, x, Z, volume_shift, omega_b=omega_b)

    a = LBC_COEFFS
    poly = a[0] + a[1] * rho_r + a[2] * rho_r**2 + a[3] * rho_r**3 + a[4] * rho_r**4
    return mu_atm + (poly**4 + a[5]) / xi_mix


class CompositionalPropertyModel(PropertyModel):
    """MRST-style property model for compositional phase properties.

    This mirrors the small public surface of MRST's
    ``CompositionalPropertyModel`` while delegating the formulas to the native
    PyMRST property helpers.
    """

    def __init__(self, fluid: Mixture | None = None, volumeShift: Optional[np.ndarray] = None, **kwargs):
        if fluid is None:
            fluid = kwargs.pop("mixture", None)
        if "volume_shift" in kwargs:
            if volumeShift is not None:
                raise TypeError("pass only one of volumeShift and volume_shift")
            volumeShift = kwargs.pop("volume_shift")
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unexpected keyword argument(s): {unknown}")
        super().__init__(fluid)
        self.volumeShift = None if volumeShift is None else np.asarray(volumeShift, float)

    @property
    def volume_shift(self) -> Optional[np.ndarray]:
        """Pythonic alias for MRST's ``volumeShift`` property."""
        return self.volumeShift

    @volume_shift.setter
    def volume_shift(self, value: Optional[np.ndarray]) -> None:
        self.volumeShift = None if value is None else np.asarray(value, float)

    def _require_fluid(self) -> Mixture:
        if self.fluid is None:
            raise ValueError("CompositionalPropertyModel.fluid must be set before evaluation")
        return self.fluid

    @staticmethod
    def _omega_b(EOS: Any) -> float:
        return float(getattr(EOS, "omegaB", OMEGA_B))

    def computeMolarDensity(self, EOS, p, x, Z, T, isLiquid=True):
        """MRST-compatible molar density method."""
        return molar_density(
            self._require_fluid(),
            p,
            T,
            x,
            Z,
            self.volumeShift,
            omega_b=self._omega_b(EOS),
        )

    def computeDensity(self, EOS, p, x, Z, T, isLiquid=True):
        """MRST-compatible mass density method."""
        return mass_density(
            self._require_fluid(),
            p,
            T,
            x,
            Z,
            self.volumeShift,
            omega_b=self._omega_b(EOS),
        )

    def computeViscosity(self, EOS, p, x, Z, T, isLiquid=True):
        """MRST-compatible Lohrenz-Bray-Clark viscosity method."""
        return viscosity_lbc(
            self._require_fluid(),
            p,
            T,
            x,
            Z,
            self.volumeShift,
            omega_b=self._omega_b(EOS),
        )

    def computePseudoCriticalPhaseProperties(self, x):
        """Return MRST's ``[P_pc, T_pc, Vc, mw]`` mixture properties."""
        mix = self._require_fluid()
        return (
            _mix(x, mix.Pcrit),
            _mix(x, mix.Tcrit),
            _mix(x, mix.Vcrit),
            _mix(x, mix.molarMass),
        )


__all__ = ["CompositionalPropertyModel", "mass_density", "molar_density", "viscosity_lbc"]
