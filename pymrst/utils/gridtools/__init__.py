"""Grid queries and mappings (MRST ``utils/gridtools``).

This package is for functions that ask something *about a grid*:
`boundary_face_indices` selects faces on a named side, `boundary_faces` finds
the boundary of a cell subset.

`rldecode`, `rlencode`, `mcolon` and `block_diag_index` used to live here and
now sit at `pymrst/utils/`, which is where MRST keeps them - they are index
arithmetic, not grid queries, and nothing about them needs a grid. They are
still re-exported here so existing imports keep working, but new code should
take them from `pymrst.utils`.
"""

from pymrst.utils.gridtools.add_bounding_box_fields import (
    add_bounding_box_fields,
)
from pymrst.utils.gridtools.create_augmented_grid import (
    create_augmented_grid,
    create_augmented_grid_tedfm,
)
from pymrst.utils.gridtools.compare_grids import compare_grids
from pymrst.utils.gridtools.compute_grid_slice_polygons import (
    compute_grid_slice_polygons,
)
from pymrst.utils.gridtools.translate_grid import translate_grid
from pymrst.utils.gridtools.compute_vertical_grid_intersection import (
    compute_vertical_grid_intersection,
)
from pymrst.utils.gridtools.make_layered_horizon_grid import (
    make_layered_horizon_grid,
)
from pymrst.utils.gridtools.convert_horizons_to_grid import (
    convert_horizons_to_grid,
)
from pymrst.utils.gridtools.refine_grdecl_layers import (
    refine_grdecl_layers,
)
from pymrst.utils.gridtools.sort_horizons import sort_horizons
from pymrst.utils.gridtools.transform_3d_to_2d_grid import (
    transform_3d_to_2d_grid,
)
from pymrst.utils.gridtools.get_faces_close_to_segment import (
    get_faces_close_to_segment,
)
from pymrst.utils.gridtools.search_for_boundary_faces import (
    search_for_boundary_faces,
)
from pymrst.utils.gridtools.make_planar_grid import make_planar_grid
from pymrst.utils.gridtools.sample_from_box import sample_from_box
from pymrst.utils.gridtools.create_grid_mappings import (
    create_grid_mappings,
)
from pymrst.utils.gridtools.volume_by_gauss_greens import (
    volume_by_gauss_greens,
)
from pymrst.utils.gridtools.find_enclosing_cell import find_enclosing_cell
from pymrst.utils.gridtools.sort_grid import sort_grid
from pymrst.utils.gridtools.check_grid import check_grid
from pymrst.utils.gridtools.grid_logical_indices import grid_logical_indices
from pymrst.utils.gridtools.grid_add_helpers import grid_add_helpers, gridAddHelpers
from pymrst.utils.gridtools.indirection_sub import indirection_sub
from pymrst.utils.gridtools.connected_cells import connected_cells
from pymrst.utils.gridtools.get_cell_no_faces import get_cell_no_faces
from pymrst.utils.gridtools.get_connectivity_matrix import get_connectivity_matrix
from pymrst.utils.gridtools.get_neighbourship import get_neighbourship
from pymrst.utils.gridtools.grid_cell_faces import grid_cell_faces
from pymrst.utils.gridtools.grid_cell_no import grid_cell_no
from pymrst.utils.gridtools.grid_cell_nodes import grid_cell_nodes
from pymrst.utils.gridtools.grid_face_nodes import grid_face_nodes
from pymrst.utils.gridtools.neighbours_by_nodes import neighbours_by_nodes
from pymrst.utils.block_diag_index import block_diag_index
from pymrst.utils.gridtools.boundary_face_indices import boundary_face_indices
from pymrst.utils.gridtools.boundary_faces import boundary_faces
from pymrst.utils.mcolon import mcolon
from pymrst.utils.rldecode import rldecode
from pymrst.utils.rlencode import rlencode

__all__ = [
    "add_bounding_box_fields",
    "compare_grids",
    "compute_grid_slice_polygons",
    "translate_grid",
    "create_augmented_grid",
    "create_augmented_grid_tedfm",
    "compute_vertical_grid_intersection",
    "make_layered_horizon_grid",
    "convert_horizons_to_grid",
    "refine_grdecl_layers",
    "sort_horizons",
    "transform_3d_to_2d_grid",
    "get_faces_close_to_segment",
    "search_for_boundary_faces",
    "make_planar_grid",
    "sample_from_box",
    "create_grid_mappings",
    "volume_by_gauss_greens",
    "find_enclosing_cell",
    "sort_grid",
    "check_grid",
    "grid_add_helpers",
    "gridAddHelpers",
    "connected_cells",
    "get_cell_no_faces",
    "get_connectivity_matrix",
    "get_neighbourship",
    "grid_cell_faces",
    "grid_cell_no",
    "grid_cell_nodes",
    "grid_face_nodes",
    "neighbours_by_nodes",
    "boundary_face_indices",
    "boundary_faces",
    # re-exported from pymrst.utils for compatibility; see the module docstring
    "block_diag_index",
    "mcolon",
    "rldecode",
    "rlencode",
]
