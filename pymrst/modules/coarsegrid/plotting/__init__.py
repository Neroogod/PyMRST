"""Coarse-grid plotting (MRST ``modules/coarsegrid/plotting``).

`outlineCoarseGrid` and `plotBlockAndNeighbors` are not here: MRST keeps them
in its core `plotting/` directory, and PyMRST mirrors that at
`pymrst/plotting/`.

Each routine splits into a function that computes what is drawn and one that
draws it, so placement is testable without a display. A label or a block in the
wrong position renders as a perfectly readable figure that says something false.
"""

from pymrst.modules.coarsegrid.plotting.explosion_view import (
    block_displacements,
    explosion_view,
    exploded_block_grids,
)
from pymrst.modules.coarsegrid.plotting.plot_cell_numbers import (
    cell_number_labels,
    plot_cell_numbers,
)
from pymrst.modules.coarsegrid.plotting.plot_face_numbers import (
    face_number_labels,
    plot_face_numbers,
)

__all__ = [
    "block_displacements",
    "cell_number_labels",
    "exploded_block_grids",
    "explosion_view",
    "face_number_labels",
    "plot_cell_numbers",
    "plot_face_numbers",
]
