"""Shale fracture-workflow compatibility helpers.

MRST's shale module layers shale-specific examples and pEDFM helpers on top of
HFM-style fracture grids. PyMRST exposes the shared fracture workflow here so
shale examples can use the same 2D/3D EDFM kernels.
"""

from pymrst.modules.hfm import (
    EDFMGrid,
    FractureLine,
    PlanarFracture,
    build_edfm,
    build_edfm_3d,
    solve_incomp_edfm,
    solve_incomp_edfm_3d,
)
from pymrst.modules.shale.data import (
    FracturePlane as ShaleFracturePlane,
    PEDFMOperators,
    ProjectedNNC,
    ShaleGlobalGrid,
    ShaleNNC,
    ShaleRock,
    WellFractureConnections,
)
from pymrst.modules.shale.compositional_fns import (
    add_sorbed_accumulation,
    ShaleFluidCaseInfo,
    NatVarsShaleModel,
    SorbedCompositionalFluid,
    apply_gangi_velocity,
    competitive_langmuir_components,
    gas_diffusion_flux,
    langmuir_sorbed_mass,
    non_darcy_velocity,
    sorbed_accumulation,
)
from pymrst.modules.shale.examples import (
    GangiCurve,
    EXAMPLES,
    ShaleExample,
    shale_example_catalog,
    LangmuirCurve,
    Compositional3DCase,
    available_shale_datasets,
    gangi_plot,
    langmuir_plot,
    load_shale_dataset,
    run_compositional_3d_case,
    run_diffusion_case,
    run_gangi_case,
    run_sorption_case,
    setup_diffusion_case,
    setup_compositional_3d_case,
    setup_gangi_case,
    setup_sorption_case,
    StochasticFractureCase,
    setup_stochastic_fractures,
    setup_eagle_ford,
    run_eagle_ford,
    setup_eagle_ford_edfm,
    run_eagle_ford_edfm,
    setup_eagle_ford_full,
    run_eagle_ford_full,
    setup_eagle_ford_edfm_full,
    run_eagle_ford_edfm_full,
    setup_eagle_ford_accelerated,
    run_eagle_ford_accelerated,
    setup_eagle_ford_edfm_accelerated,
    run_eagle_ford_edfm_accelerated,
)
from pymrst.modules.shale.other_mrst_fns import (
    add_well_shale,
    compute_wi_shale,
    make_shale_rock,
    well_fracture_nncs_3d,
)
from pymrst.modules.shale.hfm_fns import (
    assemble_shale_global_grid,
    edfm_shale_grid,
    fracture_fracture_shale_nncs_3d,
    fracture_matrix_shale_nncs_3d,
    setup_shale_edfm_operators_tpfa,
)
from pymrst.modules.shale.pedfm_nnc import (
    projected_matrix_fracture_nncs_3d,
    setup_pedfm_operators_tpfa,
    transmissibility_multiplier_pedfm,
)
from pymrst.modules.shale.spe201243pa import (
    explicit_stencil,
    explicit_stencil_3_fracs,
    load_cmg_timesteps,
    multistage_planar_nonplanar_hf,
    structured_grid_fracture_on_face,
    setup_fig_10_to_13_tene_low_kf,
    run_fig_10_to_13_tene_low_kf,
    setup_fig_10_to_13_tene_high_kf,
    run_fig_10_to_13_tene_high_kf,
    setup_fig_15_17_18_validation_case_2,
    run_fig_15_17_18_validation_case_2,
    setup_fig_16_validation_case_2_structured,
    run_fig_16_validation_case_2_structured,
    setup_fig_21_22_23_bakken_edfm,
    run_fig_21_22_23_bakken_edfm,
    setup_fig_21_22_23_bakken_pedfm,
    run_fig_21_22_23_bakken_pedfm,
)
from pymrst.modules.shale.utils import (
    create_multistage_hfs,
    gangi_fn,
    get_shale_comp_fluid_case,
    mesh_hf_system,
    FractureSystemPlot,
    plot_frac_system,
    plotFracSystem,
    plot_frac_system_nf,
    plotfracSystem_NF,
    points_to_cell_indices,
    process_stochastic_fractures,
)


def build_shale_edfm(*args, **kwargs):
    """Build a shale pEDFM-compatible 2D fracture grid."""
    return build_edfm(*args, **kwargs)


def build_shale_edfm_3d(*args, **kwargs):
    """Build a shale pEDFM-compatible 3D planar fracture grid."""
    return build_edfm_3d(*args, **kwargs)


__all__ = [
    "EDFMGrid", "FractureLine", "PlanarFracture", "build_shale_edfm",
    "build_shale_edfm_3d", "solve_incomp_edfm", "solve_incomp_edfm_3d",
    "PEDFMOperators", "ProjectedNNC", "ShaleFracturePlane", "ShaleGlobalGrid", "ShaleNNC",
    "ShaleRock",
    "WellFractureConnections", "compute_wi_shale",
    "add_well_shale",
    "assemble_shale_global_grid", "edfm_shale_grid",
    "fracture_fracture_shale_nncs_3d",
    "fracture_matrix_shale_nncs_3d",
    "setup_shale_edfm_operators_tpfa",
    "create_multistage_hfs", "gangi_fn",
    "make_shale_rock", "mesh_hf_system", "points_to_cell_indices",
    "FractureSystemPlot", "plot_frac_system", "plotFracSystem",
    "plot_frac_system_nf", "plotfracSystem_NF",
    "process_stochastic_fractures", "SorbedCompositionalFluid",
    "ShaleFluidCaseInfo", "get_shale_comp_fluid_case",
    "NatVarsShaleModel", "add_sorbed_accumulation", "apply_gangi_velocity",
    "competitive_langmuir_components", "gas_diffusion_flux",
    "langmuir_sorbed_mass", "non_darcy_velocity", "sorbed_accumulation",
    "projected_matrix_fracture_nncs_3d",
    "available_shale_datasets", "load_shale_dataset", "load_cmg_timesteps",
    "explicit_stencil", "explicit_stencil_3_fracs",
    "multistage_planar_nonplanar_hf", "structured_grid_fracture_on_face",
    "setup_fig_10_to_13_tene_low_kf", "run_fig_10_to_13_tene_low_kf",
    "setup_fig_10_to_13_tene_high_kf", "run_fig_10_to_13_tene_high_kf",
    "setup_fig_15_17_18_validation_case_2",
    "run_fig_15_17_18_validation_case_2",
    "setup_fig_16_validation_case_2_structured",
    "run_fig_16_validation_case_2_structured",
    "setup_fig_21_22_23_bakken_edfm", "run_fig_21_22_23_bakken_edfm",
    "setup_fig_21_22_23_bakken_pedfm", "run_fig_21_22_23_bakken_pedfm",
    "GangiCurve", "gangi_plot", "LangmuirCurve", "langmuir_plot",
    "EXAMPLES", "ShaleExample", "shale_example_catalog",
    "Compositional3DCase", "setup_compositional_3d_case", "run_compositional_3d_case",
    "setup_diffusion_case", "run_diffusion_case",
    "setup_gangi_case", "run_gangi_case",
    "setup_sorption_case", "run_sorption_case",
    "StochasticFractureCase", "setup_stochastic_fractures",
    "setup_eagle_ford", "run_eagle_ford",
    "setup_eagle_ford_edfm", "run_eagle_ford_edfm",
    "setup_eagle_ford_full", "run_eagle_ford_full",
    "setup_eagle_ford_edfm_full", "run_eagle_ford_edfm_full",
    "setup_eagle_ford_accelerated", "run_eagle_ford_accelerated",
    "setup_eagle_ford_edfm_accelerated", "run_eagle_ford_edfm_accelerated",
    "setup_pedfm_operators_tpfa",
    "transmissibility_multiplier_pedfm",
    "well_fracture_nncs_3d",
]
