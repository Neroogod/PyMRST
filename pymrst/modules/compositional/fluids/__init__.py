"""MRST `modules/compositional/fluids` compatibility layout."""

from pymrst.modules.compositional.fluids.compositional_mixture import (
    FlashResult,
    Mixture,
    flash,
    fugacity_coefficients,
    rachford_rice,
    simple_mixture,
    wilson_k,
)
from pymrst.modules.compositional.fluids.black_oil_property_model import BlackOilPropertyModel
from pymrst.modules.compositional.fluids.property_model import PropertyModel
from pymrst.modules.compositional.fluids.compositional_property_model import (
    CompositionalPropertyModel,
    mass_density,
    molar_density,
    viscosity_lbc,
)
from pymrst.modules.compositional.fluids.fluid_cases import (
    FluidCaseInfo,
    get_benchmark_mixture,
    get_compositional_fluid_case,
    get_compositional_mixture_case,
)

__all__ = [
    "BlackOilPropertyModel",
    "PropertyModel",
    "Mixture", "simple_mixture", "wilson_k", "rachford_rice",
    "fugacity_coefficients", "flash", "FlashResult",
    "CompositionalPropertyModel", "mass_density", "molar_density", "viscosity_lbc",
    "FluidCaseInfo", "get_benchmark_mixture", "get_compositional_fluid_case",
    "get_compositional_mixture_case",
    "FLUIDS", "MRST_CONTENTS_FLUIDS", "list_fluids", "list_mrst_contents_fluids",
    "TableCompositionalMixture", "table_compositional_mixture",
    "CoolPropsCompositionalMixture", "coolprops_compositional_mixture",
]

from pymrst.modules.compositional.fluids.contents import (
    FLUIDS,
    MRST_CONTENTS_FLUIDS,
    list_fluids,
    list_mrst_contents_fluids,
)

from pymrst.modules.compositional.fluids.table_compositional_mixture import (
    TableCompositionalMixture,
    table_compositional_mixture,
)
from pymrst.modules.compositional.fluids.coolprops_compositional_mixture import (
    CoolPropsCompositionalMixture,
    coolprops_compositional_mixture,
)
