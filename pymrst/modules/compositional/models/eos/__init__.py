"""MRST `models/eos` compatibility layout."""

from pymrst.modules.compositional.models.eos.equation_of_state_model import (
    EquationOfStateModel,
    equations_equilibrium,
    flash_ad,
    phase_compressibility_ad,
    phase_fraction_derivatives,
)
from pymrst.modules.compositional.models.eos.equilibrium_constant_model import (
    EquilibriumConstantFlashResult,
    EquilibriumConstantModel,
)

__all__ = [
    "EquationOfStateModel",
    "EquilibriumConstantModel",
    "EquilibriumConstantFlashResult",
    "phase_fraction_derivatives", "flash_ad", "equations_equilibrium",
    "phase_compressibility_ad",
]
