"""MRST-shaped HFM utility landing modules.

`build_edfm` is imported from `hfm.edfm2d`, where it is defined, rather than
from `compute_effective_trans`, which only imports `EDFMGrid` for its own
annotations. The old import asked `compute_effective_trans` for a name it
never had, so **this package could not be imported at all** - and nothing
noticed, because the sweep ran each module as a script and a script never
executes its package's `__init__`. Running them as modules is what surfaced
it.
"""

from pymrst.modules.hfm.edfm2d import EDFMGrid, build_edfm
from pymrst.modules.hfm.utils.get_avg_frac_dist_2d import (
    FractureLine,
    avg_frac_distance,
    get_avg_frac_dist_2d,
)
from pymrst.modules.hfm.utils.compute_effective_trans import (
    compute_effective_trans,
    nnc_effective_trans,
)

__all__ = ["EDFMGrid", "FractureLine", "avg_frac_distance", "build_edfm",
           "compute_effective_trans", "get_avg_frac_dist_2d",
           "nnc_effective_trans"]
