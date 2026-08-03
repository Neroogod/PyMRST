"""Coarse grids from partitions of a fine grid (MRST ``modules/coarsegrid``).

The module builds a coarse grid from a partition that assigns every fine cell
to a block, and provides the partition constructors and coarsening helpers that
upscaling and multiscale methods are built on.

Complete against MRST 2024a: 34 of 37 executable files, 3 excluded with reasons,
323 self-tests, and three MATLAB validators covering 13 cases. Two things carry
a caveat rather than a tick, both stated where they apply: `partition_metis`
needs the optional `pymetis` extra (`pip install pymrst[metis]`), and the SAIGUP
example runs on a synthetic model because its dataset is not installed.

A coarse grid here is a topology, as it is in MRST. `CoarseGrid.to_grid` gives
it as an ordinary `Grid` so `compute_trans` and `incomp_tpfa` accept it
directly; that grid has no nodes, because a block is defined by the cells it
holds rather than by a polygon.

This is the canonical location. Coarse-grid material previously lived in two
other places, both of which now re-export from here:

* ``modules/upscaling/coarsegrid`` held two of these files;
* ``modules/upscaling/upscale.py`` defined its own copies and a second
  ``CoarseGrid`` class.

The three agreed numerically, but the duplicate classes were a real hazard
rather than a tidiness question. ``upscaling/flow`` annotated the same
parameter with two different ``CoarseGrid`` types across sibling functions:
``upscale_poro`` used the one from ``upscale.py`` while ``upscale_rock`` and
``upscale_perm_fixed`` used the one from ``coarsegrid``. An object from either
path fails ``isinstance`` against the other, so nothing enforced the mismatch
only because Python does not check annotations at runtime.

See ``coarsegrid_parity.md`` for status and ``coarsegrid_implementation.md``
for the porting plan.
"""

from pymrst.modules.coarsegrid.cell_partition_to_face_partition import (
    cell_partition_to_face_partition,
)
from pymrst.modules.coarsegrid.compress_partition import compress_partition
from pymrst.modules.coarsegrid.generate_coarse_grid import (
    CoarseGrid,
    generate_coarse_grid,
)
from pymrst.modules.coarsegrid.partition_cart_grid import partition_cart_grid
from pymrst.modules.coarsegrid.partition_layers import partition_layers
from pymrst.modules.coarsegrid.partition_metis import (
    connection_matrix,
    partition_metis,
)
from pymrst.modules.coarsegrid.partition_tensor import partition_tensor
from pymrst.modules.coarsegrid.partition_ui import lb_lin_dist, partition_ui
from pymrst.modules.coarsegrid.process_face_partition import process_face_partition
from pymrst.modules.coarsegrid.process_partition import process_partition
from pymrst.modules.coarsegrid.refine_near_well import (
    refine_near_well,
    refine_partition,
)
from pymrst.modules.coarsegrid.sort_sub_edges import sort_sub_edges
from pymrst.modules.coarsegrid.utils import (
    HAVE_METIS,
    call_metis_matrix,
    coarse_data_to_fine,
    coarsen_bc,
    coarsen_flux,
    coarsen_geometry,
    fine_to_coarse_sign,
    invert_partition,
    is_refinement_of,
    is_same_partition,
    metis_graph,
    same_grouping,
)

__all__ = [
    "CoarseGrid",
    "HAVE_METIS",
    "call_metis_matrix",
    "cell_partition_to_face_partition",
    "coarse_data_to_fine",
    "coarsen_bc",
    "coarsen_flux",
    "coarsen_geometry",
    "compress_partition",
    "connection_matrix",
    "fine_to_coarse_sign",
    "generate_coarse_grid",
    "invert_partition",
    "is_refinement_of",
    "is_same_partition",
    "lb_lin_dist",
    "metis_graph",
    "partition_cart_grid",
    "partition_layers",
    "partition_metis",
    "partition_tensor",
    "partition_ui",
    "process_face_partition",
    "process_partition",
    "refine_near_well",
    "refine_partition",
    "same_grouping",
    "sort_sub_edges",
]
