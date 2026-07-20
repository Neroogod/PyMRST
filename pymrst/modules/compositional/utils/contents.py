"""Catalog for MRST compositional utility ports."""

MRST_CONTENTS_UTILS = (
    ("checkComponentMassBalance", "Check mass balance of a simulator run and print to screen"),
    ("coolPropFluidsStructs", "Fluids taken from CoolProp. Used for lookup of properties."),
    ("cubicPositive", "Straightforward implementation of a cubic root solver for vectorized"),
    ("ensureMinimumFraction", "Set a minimum value on a composition matrix"),
    ("equationsCompositional", "Overall composition fully-implicit equations"),
    ("estimateEquilibriumWilson", "Estimate equilibrium constant for a given pressure and temperature"),
    ("expandMatrixToCell", "Expand a matrix into cell arrays. Typical usage: Converting state"),
    ("FastAD", "A very limited AD class for quick EOS assembly"),
    ("formatMassString", "Small utility which returns a human readable string from mass."),
    ("getComponentsTwoPhaseSimpleWater", "Undocumented Utility Function"),
    ("getDefaultFlashNonLinearSolver", "Get default nonlinear solver for flash problems."),
    ("getEOSComponent", "Undocumented Utility Function"),
    ("getImpesWeightsOverallComposition", "Undocumented Utility Function"),
    ("getNonUnitMassFraction", "Internal utility. Intentionally undocumented."),
    ("getNonUnitMoleFraction", "Internal utility. Intentionally undocumented."),
    ("getPartialVolumes", "Undocumented Utility Function"),
    ("initCompositionalState", "Initialize a compositional state given initial composition"),
    ("initDeckEOSModel", "Set up a EquationOfState model from a parsed deck"),
    ("initVariablesFastAD", "Initialise FastAD objects used internally in some compositional code"),
    ("mrstCubic", "Straightforward implementation of a cubic root solver for vectorized"),
    ("newtonFugacityEquilibrium", "Single step of the newton update for flash equations"),
    ("phaseStabilityTest", "Perform a phase stability test for a mixture"),
    ("solveRachfordRiceVLE", "Solve Rachford Rice equations to find liquid and vapor"),
    ("standaloneFlash", "Utility for flashing without explicitly forming a state"),
    ("validateCompositionalForces", "Undocumented Utility Function"),
)

UTILS = {
    "checkComponentMassBalance": {"python_module": "check_component_mass_balance", "status": "implemented_diagnostic_core"},
    "coolPropFluidsStructs": {"python_module": "cool_prop_fluids_structs", "status": "implemented"},
    "cubicPositive": {"python_module": "cubic_positive", "status": "implemented"},
    "ensureMinimumFraction": {"python_module": "ensure_minimum_fraction", "status": "implemented"},
    "equationsCompositional": {"python_module": "equations_compositional", "status": "implemented_primary_path"},
    "estimateEquilibriumWilson": {"python_module": "estimate_equilibrium_wilson", "status": "implemented"},
    "expandMatrixToCell": {"python_module": "expand_matrix_to_cell", "status": "implemented"},
    "FastAD": {"python_module": "fast_ad", "status": "implemented"},
    "formatMassString": {"python_module": "format_mass_string", "status": "implemented"},
    "getComponentsTwoPhaseSimpleWater": {"python_module": "get_components_two_phase_simple_water", "status": "implemented"},
    "getDefaultFlashNonLinearSolver": {"python_module": "get_default_flash_nonlinear_solver", "status": "implemented"},
    "getEOSComponent": {"python_module": "get_eos_component", "status": "implemented"},
    "getImpesWeightsOverallComposition": {"python_module": "get_impes_weights_overall_composition", "status": "implemented_ad_accumulation_weights"},
    "getNonUnitMassFraction": {"python_module": "get_non_unit_mass_fraction", "status": "implemented"},
    "getNonUnitMoleFraction": {"python_module": "get_non_unit_mole_fraction", "status": "implemented"},
    "getPartialVolumes": {"python_module": "get_partial_volumes", "status": "implemented_eos_finite_difference"},
    "initCompositionalState": {"python_module": "init_compositional_state", "status": "implemented"},
    "initDeckEOSModel": {"python_module": "init_deck_eos_model", "status": "implemented"},
    "initVariablesFastAD": {"python_module": "init_variables_fast_ad", "status": "implemented"},
    "mrstCubic": {"python_module": "mrst_cubic", "status": "implemented"},
    "newtonFugacityEquilibrium": {"python_module": "newton_fugacity_equilibrium", "status": "implemented_newton_flash_step"},
    "phaseStabilityTest": {"python_module": "phase_stability_test", "status": "implemented"},
    "solveRachfordRiceVLE": {"python_module": "solve_rachford_rice_vle", "status": "implemented"},
    "standaloneFlash": {"python_module": "standalone_flash", "status": "implemented"},
    "validateCompositionalForces": {"python_module": "validate_compositional_forces", "status": "implemented_well_rhos_validation"},
}


def list_mrst_contents_utils() -> tuple[tuple[str, str], ...]:
    """Return exact entries listed by MRST's utils/Contents.m."""
    return tuple(tuple(item) for item in MRST_CONTENTS_UTILS)


def list_utils() -> dict[str, dict[str, str | None]]:
    """Return MRST utility names mapped to PyMRST port metadata."""
    return {name: dict(meta) for name, meta in UTILS.items()}


__all__ = ["MRST_CONTENTS_UTILS", "UTILS", "list_mrst_contents_utils", "list_utils"]
