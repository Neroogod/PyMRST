"""Shale-specific ports of modified MRST utility owners."""

from pymrst.modules.shale.other_mrst_fns.add_well_shale import add_well_shale
from pymrst.modules.shale.other_mrst_fns.compute_wi_shale import compute_wi_shale
from pymrst.modules.shale.other_mrst_fns.make_shale_rock import make_shale_rock
from pymrst.modules.shale.other_mrst_fns.well_fracture_nncs_3d import well_fracture_nncs_3d

__all__ = [
    "add_well_shale",
    "compute_wi_shale",
    "make_shale_rock",
    "well_fracture_nncs_3d",
]
