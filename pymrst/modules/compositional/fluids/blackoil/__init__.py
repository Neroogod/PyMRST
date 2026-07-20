"""Black-oil to compositional conversion helpers (MRST layout)."""

from pymrst.modules.compositional.fluids.blackoil.black_oil_to_mass_fraction import (
    BlackOilMassFractions,
    black_oil_to_mass_fraction,
)
from pymrst.modules.compositional.fluids.blackoil.convert_black_oil_state_to_compositional import (
    convert_black_oil_state_to_compositional,
)
from pymrst.modules.compositional.fluids.blackoil.convert_black_oil_model_to_compositional_model import (
    BlackOilCompositionalModel,
    convert_black_oil_model_to_compositional_model,
    pseudo_blackoil_mixture,
)
from pymrst.modules.compositional.fluids.blackoil.convert_rs_rv_to_k_value import (
    BlackOilPropertyTables,
    KValueTables,
    RSRVKValueProperties,
    convert_rs_rv_to_k_value,
)
from pymrst.modules.compositional.fluids.blackoil.convert_rsrv_to_k_value import (
    convertRSRVtoKValue,
    convert_rsrv_to_k_value,
)
from pymrst.modules.compositional.fluids.blackoil.kvalue_model import (
    BlackOilPropertyModel,
    EquilibriumConstantModel,
    KValueFlashResult,
)

__all__ = [
    "BlackOilMassFractions",
    "black_oil_to_mass_fraction",
    "BlackOilCompositionalModel",
    "BlackOilPropertyModel",
    "BlackOilPropertyTables",
    "EquilibriumConstantModel",
    "KValueFlashResult",
    "KValueTables",
    "RSRVKValueProperties",
    "convert_black_oil_state_to_compositional",
    "convert_black_oil_model_to_compositional_model",
    "convert_rs_rv_to_k_value",
    "convert_rsrv_to_k_value",
    "convertRSRVtoKValue",
    "pseudo_blackoil_mixture",
]
