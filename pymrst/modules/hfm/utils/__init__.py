"""MRST-shaped HFM utility landing modules."""

from pymrst.modules.hfm.utils.compute_effective_trans import EDFMGrid, build_edfm
from pymrst.modules.hfm.utils.get_avg_frac_dist_2d import (
    FractureLine,
    avg_frac_distance,
)

__all__ = ["EDFMGrid", "FractureLine", "avg_frac_distance", "build_edfm"]
