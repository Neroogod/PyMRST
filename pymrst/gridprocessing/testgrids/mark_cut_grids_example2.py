"""Random multi-cut counterpart of MRST ``markCutGridsExample2``."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry, mark_cut_grids
from pymrst.gridprocessing.slice_grid import slice_grid


def mark_cut_grids_example2(size=5, cuts=5, *, seed=0):
    rng = np.random.default_rng(seed)
    grid = compute_geometry(cart_grid([size]*3, [20.0]*3))
    cut_faces = []
    for _ in range(int(cuts)):
        old_cells = grid.cells.num
        grid = slice_grid(
            grid, 10 + 18*rng.random(3) - 9,
            normal=20*rng.random(3) - 10,
        )
        added = np.flatnonzero(
            np.all(grid.faces.neighbors >= 0, axis=1)
            & (np.max(grid.faces.neighbors, axis=1) >= old_cells)
        )
        cut_faces.extend(added.tolist())
        grid = compute_geometry(grid)
        grid.check()
    status = np.zeros(grid.faces.num, dtype=np.int64)
    status[np.asarray(cut_faces, dtype=np.int64)] = 3
    return grid, mark_cut_grids(grid, status)


markCutGridsExample2 = mark_cut_grids_example2
