"""Non-graphical counterpart of MRST's ``cutCellGrids`` stress script."""

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.gridprocessing.slice_grid import slice_grid


def cut_cell_grids(iterations=10, *, seed=0):
    """Repeatedly cut a cube and return topology/volume diagnostics."""
    rng = np.random.default_rng(seed)
    grid = compute_geometry(cart_grid([1, 1, 1], [20.0, 20.0, 20.0]))
    initial_volume = float(grid.cells.volumes.sum())
    for _ in range(int(iterations)):
        point = 10 + 18*rng.random(3) - 9
        normal = 20*rng.random(3) - 10
        grid = slice_grid(grid, point, normal=normal)
        grid = compute_geometry(grid)
        grid.check()
    return {
        "grid": grid,
        "initial_volume": initial_volume,
        "final_volume": float(grid.cells.volumes.sum()),
        "cells": grid.cells.num,
    }


cutCellGrids = cut_cell_grids
