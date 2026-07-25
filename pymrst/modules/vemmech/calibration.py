"""Backward-compatible import path for VEM mechanics calibration utilities."""

from pymrst.modules.vemmech.ad_version.calibration.calibrate_model import (
    _objective_match_value_from_stress,
    BoundaryInfo,
    CalibrationOptions,
    ElasticCalibrationBC,
    OptimizationHistory,
    calculate_stress,
    calibrate_model,
    gaussian_objfun,
    identify_boundary_elements,
    objective_control_gradient,
    objective_displacement_gradient,
    objective_stress_gradient,
    optimize_mech,
    optimize_sr1,
    produce_syndata,
    set_bcfun,
    set_material_fun,
    setup_objective_fun,
    sigma_to_smax_smin_theta,
    theta_diff,
)

# `_objective_match_value_from_stress` is private by convention, and is
# re-exported for the same reason as above: this module preserves the old
# import path, and that path included it.
__all__ = [
    "_objective_match_value_from_stress",
    "BoundaryInfo",
    "CalibrationOptions",
    "ElasticCalibrationBC",
    "OptimizationHistory",
    "calculate_stress",
    "calibrate_model",
    "gaussian_objfun",
    "identify_boundary_elements",
    "objective_control_gradient",
    "objective_displacement_gradient",
    "objective_stress_gradient",
    "optimize_mech",
    "optimize_sr1",
    "produce_syndata",
    "set_bcfun",
    "set_material_fun",
    "setup_objective_fun",
    "sigma_to_smax_smin_theta",
    "theta_diff",
]
