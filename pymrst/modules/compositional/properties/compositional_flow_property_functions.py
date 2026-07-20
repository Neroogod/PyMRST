"""Aggregate compositional flow-property state functions.

MRST 2024a lists ``CompositionalFlowPropertyFunctions`` in
``modules/compositional/properties/Contents.m`` even though the distributed tree
has no same-named MATLAB class file.  This PyMRST module provides that public
landing surface by bundling the implemented liquid/vapour compositional state
functions: phase mole/mass fractions, Z factors, densities, viscosities,
fugacities and Peng-Robinson mixing coefficients.
"""

from __future__ import annotations

from dataclasses import dataclass

from pymrst.modules.compositional.eos import Mixture
from pymrst.modules.compositional.properties.component_phase_mass_fractions_lv import (
    component_phase_mass_fractions_lv,
)
from pymrst.modules.compositional.properties.component_phase_mole_fractions_lv import (
    component_phase_mole_fractions_lv,
)
from pymrst.modules.compositional.properties.compositional_density import compositional_density
from pymrst.modules.compositional.properties.compositional_viscosity_lv import compositional_viscosity_lv
from pymrst.modules.compositional.properties.fugacity_lv import fugacity_lv
from pymrst.modules.compositional.properties.phase_compressibility_factors_lv import (
    phase_compressibility_factors_lv,
)
from pymrst.modules.compositional.properties.phase_mixing_coefficients_lv import (
    phase_mixing_coefficients_lv,
)


@dataclass
class CompositionalFlowPropertyFunctions:
    """MRST Contents-name aggregate for compositional flow properties."""

    mix: Mixture
    volume_shift: object = None
    nph: int = 2

    def evaluate(self, p, T, x, y, Z_L=None, Z_V=None) -> dict:
        """Evaluate the liquid/vapour property bundle.

        If ``Z_L``/``Z_V`` are omitted they are computed from the EOS using the
        phase compositions.  The returned mapping uses explicit MRST-style names
        so notebooks and exact example ports can request the whole flow-property
        surface without remembering each PyMRST module name.
        """
        if Z_L is None or Z_V is None:
            Z_L, Z_V = phase_compressibility_factors_lv(self.mix, p, T, x, y)
        mole_fractions = component_phase_mole_fractions_lv(x, y, self.nph)
        mass_fractions = component_phase_mass_fractions_lv(self.mix, x, y, self.nph)
        density = compositional_density(self.mix, p, T, x, y, Z_L, Z_V, self.volume_shift)
        viscosity = compositional_viscosity_lv(self.mix, p, T, x, y, Z_L, Z_V, self.volume_shift)
        fugacity = fugacity_lv(self.mix, p, T, x, y)
        mixing = phase_mixing_coefficients_lv(self.mix, p, T, x, y)
        return {
            "ComponentPhaseMoleFractionsLV": mole_fractions,
            "ComponentPhaseMassFractionsLV": mass_fractions,
            "PhaseCompressibilityFactorsLV": (Z_L, Z_V),
            "CompositionalDensity": density,
            "CompositionalViscosityLV": viscosity,
            "FugacityLV": fugacity,
            "PhaseMixingCoefficientsLV": mixing,
            "mole_fractions": mole_fractions,
            "mass_fractions": mass_fractions,
            "compressibility_factors": (Z_L, Z_V),
            "density": density,
            "viscosity": viscosity,
            "fugacity": fugacity,
            "mixing_coefficients": mixing,
        }

    __call__ = evaluate


def compositional_flow_property_functions(mix: Mixture, *, volume_shift=None, nph: int = 2):
    """Construct a ``CompositionalFlowPropertyFunctions`` aggregate."""
    return CompositionalFlowPropertyFunctions(mix, volume_shift=volume_shift, nph=nph)


__all__ = ["CompositionalFlowPropertyFunctions", "compositional_flow_property_functions"]
