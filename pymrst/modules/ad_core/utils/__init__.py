"""MRST-style utility entry points for `ad_core` (`ad-core/utils/`)."""

from pymrst.modules.ad_core.utils.operators import Operators, setup_operators_tpfa
from pymrst.modules.ad_core.utils.boundary_conditions import (
    add_bc_and_sources,
    boundary_reservoir_fluxes,
    source_reservoir_fluxes,
)
from pymrst.modules.ad_core.utils.convergence import (
    cnv_mb_convergence,
    cnv_mb_values,
    formation_volume_factor_avg,
    well_control_residual,
)
from pymrst.modules.ad_core.utils.schedule import (
    Control,
    Schedule,
    combine_schedules,
    compress_schedule,
    refine_schedule,
    simple_schedule,
    subset_schedule,
)
from pymrst.modules.ad_core.utils.cfl import (
    estimate_saturation_cfl,
    fractional_flow_derivative,
)
from pymrst.modules.ad_core.utils.phase_densities_to_bfactor import (
    phase_densities_to_bfactor,
)
from pymrst.modules.ad_core.utils.interpolate_idw import interpolate_idw
from pymrst.modules.ad_core.utils.fast_interp_table import fast_interp_table
from pymrst.modules.ad_core.utils.split_face_cell_value import split_face_cell_value
from pymrst.modules.ad_core.utils.merge_ordered_arrays import merge_ordered_arrays
from pymrst.modules.ad_core.utils.convert_reservoir_fluxes_to_surface import (
    convert_reservoir_fluxes_to_surface,
)
from pymrst.modules.ad_core.utils.select_model_from_deck import (
    deck_model_choice,
    select_model_from_deck,
)
from pymrst.modules.ad_core.utils.standalone_solve import standalone_solve
from pymrst.modules.ad_core.utils.get_well_output import get_well_output
from pymrst.modules.ad_core.utils.convert_deck_schedule_to_mrst import (
    convert_deck_schedule_to_mrst,
    convertDeckScheduleToMRST,
)
from pymrst.modules.ad_core.utils.make_schedule_consistent import (
    make_schedule_consistent,
    makeScheduleConsistent,
)
from pymrst.modules.ad_core.utils.init_well_sol_ad import (
    init_well_sol_ad,
    initWellSolAD,
)
from pymrst.modules.ad_core.utils.set_well_sign import set_well_sign, setWellSign
from pymrst.modules.ad_core.utils.assign_well_values_from_control import (
    assign_well_values_from_control,
    assignWellValuesFromControl,
)
from pymrst.modules.ad_core.utils.pad_rates_and_compi import (
    pad_rates_and_compi,
    padRatesAndCompi,
)
from pymrst.modules.ad_core.utils.well_sol_to_vector import (
    well_sol_to_vector,
    wellSolToVector,
)
from pymrst.modules.ad_core.utils.recover_vars import recover_vars, recoverVars
from pymrst.modules.ad_core.utils.expand_if_uniform import (
    expand_if_uniform,
    expandIfUniform,
)
from pymrst.modules.ad_core.utils.expand_matrix_to_cell import (
    expand_matrix_to_cell,
    expandMatrixToCell,
)
from pymrst.modules.ad_core.utils.numel_data import numel_data, numelData
from pymrst.modules.ad_core.utils.size_jac import size_jac, sizeJac
from pymrst.modules.ad_core.utils.critical_point_chop import (
    critical_point_chop,
    criticalPointChop,
)
from pymrst.modules.ad_core.utils.split_matrix_for_reduction import (
    ReductionSystem,
    split_matrix_for_reduction,
    splitMatrixForReduction,
)
from pymrst.modules.ad_core.utils.get_cell_major_reordering import (
    get_cell_major_reordering,
    getCellMajorReordering,
)
from pymrst.modules.ad_core.utils.get_perforation_to_well_mapping import (
    get_perforation_to_well_mapping,
    getPerforationToWellMapping,
)
from pymrst.modules.ad_core.utils.convert_incomp_well_sols import (
    convert_incomp_well_sols,
    convertIncompWellSols,
)
from pymrst.modules.ad_core.utils.get_report_ministeps import (
    get_report_ministeps,
    getReportMinisteps,
)

__all__ = [
    "Operators",
    "add_bc_and_sources",
    "boundary_reservoir_fluxes",
    "setup_operators_tpfa",
    "source_reservoir_fluxes",
    "cnv_mb_convergence",
    "cnv_mb_values",
    "formation_volume_factor_avg",
    "well_control_residual",
    "Control",
    "Schedule",
    "combine_schedules",
    "compress_schedule",
    "refine_schedule",
    "simple_schedule",
    "subset_schedule",
    "estimate_saturation_cfl",
    "fractional_flow_derivative",
    "phase_densities_to_bfactor",
    "interpolate_idw",
    "fast_interp_table",
    "split_face_cell_value",
    "merge_ordered_arrays",
    "convert_reservoir_fluxes_to_surface",
    "deck_model_choice",
    "select_model_from_deck",
    "standalone_solve",
    "get_well_output",
    "convert_deck_schedule_to_mrst",
    "convertDeckScheduleToMRST",
    "make_schedule_consistent",
    "makeScheduleConsistent",
    "init_well_sol_ad",
    "initWellSolAD",
    "set_well_sign",
    "setWellSign",
    "assign_well_values_from_control",
    "assignWellValuesFromControl",
    "pad_rates_and_compi",
    "padRatesAndCompi",
    "well_sol_to_vector",
    "wellSolToVector",
    "recover_vars",
    "recoverVars",
    "expand_if_uniform",
    "expandIfUniform",
    "expand_matrix_to_cell",
    "expandMatrixToCell",
    "numel_data",
    "numelData",
    "size_jac",
    "sizeJac",
    "critical_point_chop",
    "criticalPointChop",
    "ReductionSystem",
    "split_matrix_for_reduction",
    "splitMatrixForReduction",
    "get_cell_major_reordering",
    "getCellMajorReordering",
    "get_perforation_to_well_mapping",
    "getPerforationToWellMapping",
    "convert_incomp_well_sols",
    "convertIncompWellSols",
    "get_report_ministeps",
    "getReportMinisteps",
]
