"""Reproduction workflows for SPE-201243-PA."""

from pymrst.modules.shale.spe201243pa.data_loader import load_cmg_timesteps
from pymrst.modules.shale.spe201243pa.explicit_stencil import explicit_stencil, explicitStencil
from pymrst.modules.shale.spe201243pa.explicit_stencil_3_fracs import explicit_stencil_3_fracs, explicitStencil3Fracs
from pymrst.modules.shale.spe201243pa.multistage_planar_nonplanar_hf import (
    MulistagePlanarNonPlanarHF,
    multistage_planar_nonplanar_hf,
)
from pymrst.modules.shale.spe201243pa.structured_grid_fracture_on_face import (
    StructuredGridFractureonFace,
    structured_grid_fracture_on_face,
)
from pymrst.modules.shale.spe201243pa.fig_10_to_13_tene_low_kf import (
    Fig10to13TeneCaseLowKf,
    run_fig_10_to_13_tene_low_kf,
    setup_fig_10_to_13_tene_low_kf,
)
from pymrst.modules.shale.spe201243pa.fig_10_to_13_tene_high_kf import (
    Fig10to13TeneCasesHighKf,
    run_fig_10_to_13_tene_high_kf,
    setup_fig_10_to_13_tene_high_kf,
)
from pymrst.modules.shale.spe201243pa.fig_15_17_18_validation_case_2 import (
    Fig15_17_18_validationCase2,
    run_fig_15_17_18_validation_case_2,
    setup_fig_15_17_18_validation_case_2,
)
from pymrst.modules.shale.spe201243pa.fig_16_validation_case_2_structured import (
    Fig16_validationCase2_structured,
    run_fig_16_validation_case_2_structured,
    setup_fig_16_validation_case_2_structured,
)
from pymrst.modules.shale.spe201243pa.fig_21_22_23_bakken_edfm import (
    Fig21_22_23_bakken_edfm,
    run_fig_21_22_23_bakken_edfm,
    setup_fig_21_22_23_bakken_edfm,
)
from pymrst.modules.shale.spe201243pa.fig_21_22_23_bakken_pedfm import (
    Fig21_22_23_bakken_pedfm,
    run_fig_21_22_23_bakken_pedfm,
    setup_fig_21_22_23_bakken_pedfm,
)

__all__ = [
    "load_cmg_timesteps", "explicit_stencil", "explicitStencil",
    "explicit_stencil_3_fracs", "explicitStencil3Fracs",
    "multistage_planar_nonplanar_hf", "MulistagePlanarNonPlanarHF",
    "structured_grid_fracture_on_face", "StructuredGridFractureonFace",
    "setup_fig_10_to_13_tene_low_kf", "run_fig_10_to_13_tene_low_kf",
    "Fig10to13TeneCaseLowKf", "setup_fig_10_to_13_tene_high_kf",
    "run_fig_10_to_13_tene_high_kf", "Fig10to13TeneCasesHighKf",
    "setup_fig_15_17_18_validation_case_2",
    "run_fig_15_17_18_validation_case_2", "Fig15_17_18_validationCase2",
    "setup_fig_16_validation_case_2_structured",
    "run_fig_16_validation_case_2_structured",
    "Fig16_validationCase2_structured",
    "setup_fig_21_22_23_bakken_edfm", "run_fig_21_22_23_bakken_edfm",
    "Fig21_22_23_bakken_edfm", "setup_fig_21_22_23_bakken_pedfm",
    "run_fig_21_22_23_bakken_pedfm", "Fig21_22_23_bakken_pedfm",
]
