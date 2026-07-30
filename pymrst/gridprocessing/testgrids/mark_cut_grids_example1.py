"""Non-plotting counterpart of MRST ``markCutGridsExample1``."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry, mark_cut_grids
from pymrst.gridprocessing.slice_grid import slice_grid


def mark_cut_grids_example1(nx=5):
    grid = compute_geometry(cart_grid([nx, nx, nx], [1.0, 1.0, 1.0]))
    cut = slice_grid(grid, np.array([.5, .5, .5]), normal=np.array([1., 1., 0.]))
    cut = compute_geometry(cut)
    # The inserted cut faces connect an original cell to its split-off cell.
    status = np.zeros(cut.faces.num, dtype=np.int64)
    split_faces = np.all(cut.faces.neighbors >= 0, axis=1) & (
        np.max(cut.faces.neighbors, axis=1) >= grid.cells.num
    )
    status[split_faces] = 3
    return cut, mark_cut_grids(cut, status)


markCutGridsExample1 = mark_cut_grids_example1
