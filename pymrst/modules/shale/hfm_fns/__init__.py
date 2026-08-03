"""Shale EDFM gridding and operator owners built on ``hfm``."""

from pymrst.modules.shale.hfm_fns.assemble_shale_global_grid import assemble_shale_global_grid
from pymrst.modules.shale.hfm_fns.edfm_shale_grid import edfm_shale_grid
from pymrst.modules.shale.hfm_fns.fracture_matrix_shale_nncs_3d import fracture_matrix_shale_nncs_3d
from pymrst.modules.shale.hfm_fns.setup_shale_edfm_operators_tpfa import setup_shale_edfm_operators_tpfa
from pymrst.modules.shale.hfm_fns.fracture_fracture_shale_nncs_3d import fracture_fracture_shale_nncs_3d

__all__ = [
    "assemble_shale_global_grid",
    "edfm_shale_grid",
    "fracture_fracture_shale_nncs_3d",
    "fracture_matrix_shale_nncs_3d",
    "setup_shale_edfm_operators_tpfa",
]
