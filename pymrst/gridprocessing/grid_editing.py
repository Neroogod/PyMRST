"""Compatibility exports for grid editing primitives."""

from pymrst.gridprocessing.private.add_faces import add_faces
from pymrst.gridprocessing.private.empty_grid import empty_grid
from pymrst.gridprocessing.private.remove_faces import remove_faces
from pymrst.gridprocessing.remove_cells import remove_cells

__all__ = ["add_faces", "empty_grid", "remove_cells", "remove_faces"]

