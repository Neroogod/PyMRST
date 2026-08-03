"""General utilities (MRST ``utils/``).

Index arithmetic, table interpolation, flux conversion and eigen-decomposition
for many small matrices, plus the `units` and `adi` subpackages.

The layout follows MRST's: `rldecode`, `rlencode`, `mcolon` and
`block_diag_index` are at this level rather than under `gridtools`, because
none of them takes a grid.
"""

from pymrst.utils.block_diag_index import block_diag_index
from pymrst.utils.cell_dims_cg import cell_dims_cg
from pymrst.utils.dinterp_table import dinterp_table
from pymrst.utils.dinterpq1 import dinterpq1
from pymrst.utils.discspace1 import discspace1
from pymrst.utils.eigs import multi_eig, multi_symm_eig
from pymrst.utils.format_time_range import format_time_range
from pymrst.utils.geomspace import geomspace
from pymrst.utils.get_sorted_cell_nodes import (
    get_sorted_cell_nodes,
    sort_edges,
)
from pymrst.utils.incomp_hydrostatic_pressure import (
    incomp_hydrostatic_pressure,
)
from pymrst.utils.is_coarse_grid import is_coarse_grid
from pymrst.utils.numel_value import numel_value
from pymrst.utils.reduce_to_double import reduce_to_double
from pymrst.utils.subset_arithmetic import subset_minus, subset_plus
from pymrst.utils.cell_flux_to_face_flux import cell_flux_to_face_flux
from pymrst.utils.face_flux_to_cell_flux import face_flux_to_cell_flux
from pymrst.utils.face_flux_to_cell_velocity import face_flux_to_cell_velocity
from pymrst.utils.interp_table import interp_table
from pymrst.utils.mcolon import mcolon
from pymrst.utils.rldecode import rldecode
from pymrst.utils.rlencode import rlencode
from pymrst.utils.unique_stable import unique_stable

__all__ = [
    "block_diag_index",
    "cell_dims_cg",
    "dinterp_table",
    "dinterpq1",
    "discspace1",
    "format_time_range",
    "geomspace",
    "get_sorted_cell_nodes",
    "incomp_hydrostatic_pressure",
    "is_coarse_grid",
    "numel_value",
    "reduce_to_double",
    "sort_edges",
    "subset_minus",
    "subset_plus",
    "cell_flux_to_face_flux",
    "face_flux_to_cell_flux",
    "face_flux_to_cell_velocity",
    "interp_table",
    "mcolon",
    "multi_eig",
    "multi_symm_eig",
    "rldecode",
    "rlencode",
    "unique_stable",
]
