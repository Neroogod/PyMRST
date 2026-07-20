"""Compositional flow: PR equation of state and flash (MRST compositional)."""

from pymrst.modules.compositional.eos import (
    FlashResult,
    Mixture,
    flash,
    fugacity_coefficients,
    rachford_rice,
    simple_mixture,
    wilson_k,
)
from pymrst.modules.compositional.eos_ad import (
    equations_equilibrium,
    flash_ad,
    phase_compressibility_ad,
    phase_fraction_derivatives,
)
from pymrst.modules.compositional.models.eos import EquationOfStateModel
from pymrst.modules.compositional.fluids.fluid_cases import (
    FluidCaseInfo,
    get_benchmark_mixture,
    get_compositional_fluid_case,
    get_compositional_mixture_case,
)
from pymrst.modules.compositional.flow_props import (
    FlowProps,
    flow_props_from_flash,
    mass_fraction,
    mole_fraction,
    phase_saturations,
    two_phase_flow_props,
)
from pymrst.modules.compositional.backend import (
    backend_flash,
    backend_flow_props,
    backend_flow_props_ad,
    backend_natural_props,
    backend_natural_props_ad,
)
from pymrst.modules.compositional.model import (
    CompositionalState,
    FacilityGroupControl,
    OverallCompositionModel,
    setTimeDiscretization,
    set_time_discretization,
)
from pymrst.modules.compositional.facility import (
    applyDeckWellControls,
    apply_deck_well_controls,
    facilityGroupControlsFromDeckControl,
    facilityGroupControlsFromDeckSchedule,
    facility_group_controls_from_deck_control,
    facility_group_controls_from_deck_schedule,
    wellsFromDeckCompletionRecords,
    wells_from_deck_completion_records,
)
from pymrst.modules.compositional.natural_variables import (
    GenericNaturalVariablesModel,
    NaturalVariablesCompositionalModel,
    NaturalVariablesState,
)
from pymrst.modules.compositional.properties import (
    mass_density,
    molar_density,
    viscosity_lbc,
)
from pymrst.modules.compositional.fluids.compositional_property_model import (
    CompositionalPropertyModel,
)
from pymrst.modules.compositional.utils.init_compositional_state import (
    initCompositionalState,
    init_compositional_state,
)

__all__ = [
    "Mixture", "simple_mixture", "wilson_k", "rachford_rice",
    "fugacity_coefficients", "flash", "FlashResult",
    "FluidCaseInfo", "get_benchmark_mixture", "get_compositional_fluid_case",
    "get_compositional_mixture_case",
    "EquationOfStateModel", "CompositionalPropertyModel",
    "mass_density", "molar_density", "viscosity_lbc",
    "mass_fraction", "mole_fraction", "phase_saturations",
    "two_phase_flow_props", "flow_props_from_flash", "FlowProps",
    "backend_flash", "backend_flow_props", "backend_flow_props_ad",
    "backend_natural_props", "backend_natural_props_ad",
    "phase_fraction_derivatives", "flash_ad", "equations_equilibrium",
    "phase_compressibility_ad",
    "init_compositional_state", "initCompositionalState",
    "OverallCompositionModel", "CompositionalState", "FacilityGroupControl",
    "set_time_discretization", "setTimeDiscretization",
    "apply_deck_well_controls", "applyDeckWellControls",
    "facility_group_controls_from_deck_control",
    "facility_group_controls_from_deck_schedule",
    "facilityGroupControlsFromDeckControl",
    "facilityGroupControlsFromDeckSchedule",
    "wells_from_deck_completion_records", "wellsFromDeckCompletionRecords",
    "NaturalVariablesCompositionalModel", "GenericNaturalVariablesModel",
    "NaturalVariablesState",
]
