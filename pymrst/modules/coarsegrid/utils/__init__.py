"""Coarsening helpers (MRST ``modules/coarsegrid/utils``).

`generate_coarse_grid` builds topology. These take everything else across the
scale change: geometry, data, fluxes, boundary conditions, and the block-to-cell
lookup they all rest on.

MRST's `utils/callMetisMatrix.m` is the eighth file here. It is ported, but its
partitioning call needs the optional `pymetis` package; the graph it builds is
available without one.

**Import order matters below.** Every module here is named after the function it
holds, so importing the module binds the name to the *module* and a later
`from ... import coarsen_geometry` returns something that is not callable. The
submodules are imported first and the function names rebound afterwards. This
codebase has lost time to that trap three times already, most recently in
`modules/upscaling/coarsegrid`.
"""

from pymrst.modules.coarsegrid.utils import (  # noqa: F401  (bind the modules first)
    call_metis_matrix as _call_metis_matrix_module,
    coarse_data_to_fine as _coarse_data_to_fine_module,
    coarsen_bc as _coarsen_bc_module,
    coarsen_flux as _coarsen_flux_module,
    coarsen_geometry as _coarsen_geometry_module,
    fine_to_coarse_sign as _fine_to_coarse_sign_module,
    invert_partition as _invert_partition_module,
    is_same_partition as _is_same_partition_module,
)

from pymrst.modules.coarsegrid.utils.call_metis_matrix import (
    HAVE_METIS,
    call_metis_matrix,
    metis_graph,
)
from pymrst.modules.coarsegrid.utils.coarse_data_to_fine import coarse_data_to_fine
from pymrst.modules.coarsegrid.utils.coarsen_bc import coarsen_bc
from pymrst.modules.coarsegrid.utils.coarsen_flux import coarsen_flux
from pymrst.modules.coarsegrid.utils.coarsen_geometry import coarsen_geometry
from pymrst.modules.coarsegrid.utils.fine_to_coarse_sign import fine_to_coarse_sign
from pymrst.modules.coarsegrid.utils.invert_partition import invert_partition
from pymrst.modules.coarsegrid.utils.is_same_partition import (
    is_refinement_of,
    is_same_partition,
    same_grouping,
)

__all__ = [
    "HAVE_METIS",
    "call_metis_matrix",
    "coarse_data_to_fine",
    "coarsen_bc",
    "coarsen_flux",
    "coarsen_geometry",
    "fine_to_coarse_sign",
    "invert_partition",
    "is_refinement_of",
    "is_same_partition",
    "metis_graph",
    "same_grouping",
]
