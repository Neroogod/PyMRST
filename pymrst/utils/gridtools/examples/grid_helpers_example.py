"""Runnable counterpart of MRST ``gridHelpersExample.m``."""

from __future__ import annotations

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.utils.gridtools.grid_add_helpers import grid_add_helpers


def grid_helpers_example(*, shape=(10, 20, 3), plot=False):
    grid = grid_add_helpers(compute_geometry(cart_grid(list(shape))))
    cells = np.unique(
        np.array([0, min(1, grid.cells.num - 1), min(9, grid.cells.num - 1)], dtype=int)
    )
    nodes, node_pos = grid.helpers.getCellNodes(cells)
    faces, face_pos = grid.helpers.getCellFaces(cells)
    if plot:
        grid.plot.grid(cells=cells)
        grid.plot.points(grid.nodes.coords[nodes])
    return {
        "grid": grid, "cells": cells, "nodes": nodes, "node_pos": node_pos,
        "faces": faces, "face_pos": face_pos,
    }


if __name__ == "__main__":
    result = grid_helpers_example()
    assert result["nodes"].size > 0 and result["faces"].size > 0
    print("gridHelpersExample passed\n\nAll tests passed!")
