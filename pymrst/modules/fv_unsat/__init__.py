"""Unsaturated flow and poromechanics (MRST ``modules/fv-unsat``)."""

from pymrst.modules.fv_unsat.data.get_hydraulic_properties import (
    HydraulicProperties,
    get_hydraulic_properties,
)
from pymrst.modules.fv_unsat.utility.vgm_saturation import vgm_saturation
from pymrst.modules.fv_unsat.utility.vgm_theta import vgm_theta
from pymrst.modules.fv_unsat.models.model_re import RichardsModel, model_re
from pymrst.modules.fv_unsat.utility.solver_re import RichardsStep, solver_re

getHydraulicProperties = get_hydraulic_properties
vGM_saturation = vgm_saturation
vGM_theta = vgm_theta
modelRE = model_re
solverRE = solver_re

__all__ = [
    "HydraulicProperties",
    "RichardsModel",
    "RichardsStep",
    "get_hydraulic_properties",
    "getHydraulicProperties",
    "vgm_saturation",
    "vgm_theta",
    "vGM_saturation",
    "vGM_theta",
    "model_re",
    "modelRE",
    "solver_re",
    "solverRE",
]
