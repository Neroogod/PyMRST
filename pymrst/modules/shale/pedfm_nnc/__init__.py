"""Projection-based EDFM NNC algorithms."""

from pymrst.modules.shale.pedfm_nnc.projected_matrix_fracture_nncs_3d import projected_matrix_fracture_nncs_3d
from pymrst.modules.shale.pedfm_nnc.transmissibility_multiplier_pedfm import transmissibility_multiplier_pedfm
from pymrst.modules.shale.pedfm_nnc.setup_pedfm_operators_tpfa import setup_pedfm_operators_tpfa

__all__ = [
    "projected_matrix_fracture_nncs_3d",
    "setup_pedfm_operators_tpfa",
    "transmissibility_multiplier_pedfm",
]
