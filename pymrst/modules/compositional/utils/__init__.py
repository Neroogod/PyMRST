"""MRST `modules/compositional/utils` compatibility layout."""

from pymrst.modules.compositional.utils.check_component_mass_balance import (
    checkComponentMassBalance,
    check_component_mass_balance,
)
from pymrst.modules.compositional.utils.cool_prop_fluids_structs import (
    COOLPROP_FLUIDS,
    CoolPropFluid,
    coolPropFluidsStructs,
    cool_prop_fluids_structs,
)
from pymrst.modules.compositional.utils.contents import (
    MRST_CONTENTS_UTILS,
    UTILS,
    list_mrst_contents_utils,
    list_utils,
)
from pymrst.modules.compositional.utils.cubic_positive import (
    cubicPositive,
    cubic_positive,
)
from pymrst.modules.compositional.utils.equations_compositional import (
    CompositionalState,
    OverallCompositionModel,
)
from pymrst.modules.compositional.utils.estimate_equilibrium_wilson import (
    estimateEquilibriumWilson,
    estimate_equilibrium_wilson,
    wilson_k,
)
from pymrst.modules.compositional.utils.ensure_minimum_fraction import (
    ensureMinimumFraction,
    ensure_minimum_fraction,
)
from pymrst.modules.compositional.utils.expand_matrix_to_cell import (
    expandMatrixToCell,
    expand_matrix_to_cell,
)
from pymrst.modules.compositional.utils.fast_ad import (
    FastAD,
    exp as fast_ad_exp,
    fast_max,
    fast_sum,
    log as fast_ad_log,
    value as fast_ad_value,
)
from pymrst.modules.compositional.utils.format_mass_string import (
    formatMassString,
    format_mass_string,
)
from pymrst.modules.compositional.utils.get_default_flash_nonlinear_solver import (
    getDefaultFlashNonLinearSolver,
    get_default_flash_nonlinear_solver,
)
from pymrst.modules.compositional.utils.get_eos_component import (
    getEOSComponent,
    get_eos_component,
)
from pymrst.modules.compositional.utils.get_impes_weights_overall_composition import (
    getImpesWeightsOverallComposition,
    get_impes_weights_overall_composition,
)
from pymrst.modules.compositional.utils.get_flash_table import (
    getFlashTable,
    get_flash_table,
)
from pymrst.modules.compositional.utils.get_non_unit_mass_fraction import (
    getNonUnitMassFraction,
    get_non_unit_mass_fraction,
)
from pymrst.modules.compositional.utils.get_non_unit_mole_fraction import (
    getNonUnitMoleFraction,
    get_non_unit_mole_fraction,
)
from pymrst.modules.compositional.utils.get_partial_volumes import (
    getPartialVolumes,
    get_partial_volumes,
)
from pymrst.modules.compositional.utils.get_components_two_phase_simple_water import (
    getComponentsTwoPhaseSimpleWater,
    get_components_two_phase_simple_water,
)
from pymrst.modules.compositional.utils.init_compositional_state import (
    initCompositionalState,
    init_compositional_state,
)
from pymrst.modules.compositional.utils.init_deck_eos_model import (
    initDeckEOSModel,
    init_deck_eos_model,
)
from pymrst.modules.compositional.utils.init_variables_fast_ad import (
    initVariablesFastAD,
    init_variables_fast_ad,
)
from pymrst.modules.compositional.utils.mrst_cubic import (
    mrstCubic,
    mrst_cubic,
)
from pymrst.modules.compositional.utils.newton_fugacity_equilibrium import (
    newtonFugacityEquilibrium,
    newton_fugacity_equilibrium,
)
from pymrst.modules.compositional.utils.phase_stability_test import (
    phaseStabilityTest,
    phase_stability_test,
)
from pymrst.modules.compositional.utils.solve_rachford_rice_vle import (
    rachford_rice,
    solveRachfordRiceVLE,
    solve_rachford_rice_vle,
)
from pymrst.modules.compositional.utils.standalone_flash import (
    FlashResult,
    flash,
    standaloneFlash,
    standalone_flash,
)
from pymrst.modules.compositional.utils.units import (
    fromCelsius,
    fromFahrenheit,
    from_celsius,
    from_fahrenheit,
)
from pymrst.modules.compositional.utils.validate_compositional_forces import (
    validateCompositionalForces,
    validate_compositional_forces,
)

__all__ = [
    "checkComponentMassBalance", "check_component_mass_balance",
    "OverallCompositionModel", "CompositionalState",
    "MRST_CONTENTS_UTILS", "UTILS", "list_mrst_contents_utils", "list_utils",
    "COOLPROP_FLUIDS", "CoolPropFluid", "coolPropFluidsStructs", "cool_prop_fluids_structs",
    "cubicPositive", "cubic_positive", "mrstCubic", "mrst_cubic",
    "newtonFugacityEquilibrium", "newton_fugacity_equilibrium",
    "FastAD", "initVariablesFastAD", "init_variables_fast_ad",
    "fast_ad_exp", "fast_ad_log", "fast_ad_value", "fast_sum", "fast_max",
    "init_compositional_state", "initCompositionalState",
    "initDeckEOSModel", "init_deck_eos_model",
    "ensure_minimum_fraction", "ensureMinimumFraction",
    "expandMatrixToCell", "expand_matrix_to_cell",
    "format_mass_string", "formatMassString",
    "getDefaultFlashNonLinearSolver", "get_default_flash_nonlinear_solver",
    "getEOSComponent", "get_eos_component",
    "getImpesWeightsOverallComposition", "get_impes_weights_overall_composition",
    "getFlashTable", "get_flash_table",
    "get_non_unit_mass_fraction", "getNonUnitMassFraction",
    "get_non_unit_mole_fraction", "getNonUnitMoleFraction",
    "getPartialVolumes", "get_partial_volumes",
    "get_components_two_phase_simple_water", "getComponentsTwoPhaseSimpleWater",
    "wilson_k", "estimateEquilibriumWilson", "estimate_equilibrium_wilson",
    "phaseStabilityTest", "phase_stability_test",
    "rachford_rice", "solveRachfordRiceVLE", "solve_rachford_rice_vle",
    "flash", "standaloneFlash", "standalone_flash", "FlashResult",
    "fromCelsius", "from_celsius", "fromFahrenheit", "from_fahrenheit",
    "validateCompositionalForces", "validate_compositional_forces",
]
