"""Compositional property functions and MRST-style property classes."""

from typing import Any

import numpy as np

from pymrst.modules.compositional.eos import OMEGA_B
from pymrst.modules.compositional.eos import Mixture
from pymrst.utils.adi import value

R_GAS: float = 8.3144598
RANKINE: float = 5.0 / 9.0
PSIA: float = 6894.757293168361
CENTIPOISE: float = 1.0e-3
GRAM: float = 1.0e-3
LBC_COEFFS = (0.1023, 0.023364, 0.058533, -0.040758, 0.0093324, -1.0e-4)


def composition_weighted_sum(comp: Any, weights: np.ndarray) -> Any:
    """Compute `sum_i weights_i comp_i`, preserving ADI columns."""
    weights = np.asarray(weights, float)
    if isinstance(comp, (list, tuple)):
        total = comp[0] * weights[0]
        for i in range(1, len(comp)):
            total = total + comp[i] * weights[i]
        return total
    return np.asarray(comp, float) @ weights


def _mix(comp: Any, weights: np.ndarray) -> Any:
    """MRST/PyMRST-compatible alias for composition-weighted sums."""
    return composition_weighted_sum(comp, weights)


def molar_density(mix: Mixture, p, T, x, Z, volume_shift: np.ndarray | None = None) -> Any:
    """Molar density [mol/m3], equivalent to MRST `computeMolarDensity`."""
    v = (R_GAS * T) * Z / p
    if volume_shift is not None:
        beta = OMEGA_B * R_GAS * mix.Tcrit / mix.Pcrit
        v = v - composition_weighted_sum(x, np.asarray(volume_shift, float) * beta)
    return 1.0 / v


def mass_density(mix: Mixture, p, T, x, Z, volume_shift: np.ndarray | None = None) -> Any:
    """Mass density [kg/m3], equivalent to MRST `computeDensity`."""
    molar_mass = composition_weighted_sum(x, mix.molarMass)
    return molar_density(mix, p, T, x, Z, volume_shift) * molar_mass


def viscosity_lbc(mix: Mixture, p, T, x, Z, volume_shift: np.ndarray | None = None) -> Any:
    """Lohrenz-Bray-Clark viscosity [Pa s], equivalent to MRST `computeViscosity`."""
    t_val = np.asarray(value(T), dtype=float)
    m_gram = mix.molarMass / GRAM
    sqrt_m = np.sqrt(m_gram)
    tc_r = mix.Tcrit / RANKINE
    pc_psi = mix.Pcrit / PSIA
    xi = 5.4402 * tc_r ** (1.0 / 6.0) / (sqrt_m * pc_psi ** (2.0 / 3.0) * CENTIPOISE)

    num = None
    den = None
    for i in range(mix.n):
        tr = t_val / mix.Tcrit[i]
        mu_i = np.where(
            tr > 1.5,
            17.78e-5 * np.maximum(4.58 * tr - 1.67, 0.0) ** 0.625,
            34e-5 * tr ** 0.94,
        ) / xi[i]
        xi_i = x[i] if isinstance(x, (list, tuple)) else np.asarray(x, float)[:, i]
        term_num = xi_i * (mu_i * sqrt_m[i])
        term_den = xi_i * sqrt_m[i]
        num = term_num if num is None else num + term_num
        den = term_den if den is None else den + term_den
    mu_atm = num / den

    t_pc = composition_weighted_sum(x, mix.Tcrit)
    p_pc = composition_weighted_sum(x, mix.Pcrit)
    v_c = composition_weighted_sum(x, mix.Vcrit)
    m_mix = composition_weighted_sum(x, mix.molarMass)
    xi_mix = (
        5.4402 * (t_pc / RANKINE) ** (1.0 / 6.0)
        / ((m_mix / GRAM) ** 0.5 * (p_pc / PSIA) ** (2.0 / 3.0) * CENTIPOISE)
    )
    rho_r = v_c * molar_density(mix, p, T, x, Z, volume_shift)
    c = LBC_COEFFS
    poly = c[0] + c[1] * rho_r + c[2] * rho_r ** 2 + c[3] * rho_r ** 3 + c[4] * rho_r ** 4
    return mu_atm + (poly ** 4 + c[5]) / xi_mix


from pymrst.modules.compositional.properties.component_phase_mass_fractions_lv import (  # noqa: E402
    ComponentPhaseMassFractionsLV,
    component_phase_mass_fractions_lv,
)
from pymrst.modules.compositional.properties.component_phase_mole_fractions_lv import (  # noqa: E402
    ComponentPhaseMoleFractionsLV,
    component_phase_mole_fractions_lv,
)
from pymrst.modules.compositional.properties.compositional_density import (  # noqa: E402
    CompositionalDensity,
    compositional_density,
)
from pymrst.modules.compositional.properties.compositional_flow_property_functions import (  # noqa: E402
    CompositionalFlowPropertyFunctions,
    compositional_flow_property_functions,
)
from pymrst.modules.compositional.properties.compositional_viscosity_lv import (  # noqa: E402
    CompositionalViscosityLV,
    compositional_viscosity_lv,
)
from pymrst.modules.compositional.properties.fugacity_lv import FugacityLV, fugacity_lv  # noqa: E402
from pymrst.modules.compositional.properties.phase_compressibility_factors_lv import (  # noqa: E402
    PhaseCompressibilityFactorsLV,
    phase_compressibility_factors_lv,
)
from pymrst.modules.compositional.properties.phase_mixing_coefficients_lv import (  # noqa: E402
    PhaseMixingCoefficients,
    PhaseMixingCoefficientsLV,
    phase_mixing_coefficients_lv,
)
from pymrst.modules.compositional.properties.components.equation_of_state_component import (  # noqa: E402
    EquationOfStateComponent,
)

__all__ = [
    "CENTIPOISE",
    "GRAM",
    "LBC_COEFFS",
    "PSIA",
    "RANKINE",
    "R_GAS",
    "ComponentPhaseMassFractionsLV",
    "ComponentPhaseMoleFractionsLV",
    "CompositionalDensity",
    "CompositionalFlowPropertyFunctions",
    "CompositionalViscosityLV",
    "EquationOfStateComponent",
    "FugacityLV",
    "PhaseCompressibilityFactorsLV",
    "PhaseMixingCoefficients",
    "PhaseMixingCoefficientsLV",
    "_mix",
    "component_phase_mass_fractions_lv",
    "component_phase_mole_fractions_lv",
    "composition_weighted_sum",
    "compositional_density",
    "compositional_flow_property_functions",
    "compositional_viscosity_lv",
    "fugacity_lv",
    "mass_density",
    "molar_density",
    "phase_compressibility_factors_lv",
    "phase_mixing_coefficients_lv",
    "viscosity_lbc",
]
