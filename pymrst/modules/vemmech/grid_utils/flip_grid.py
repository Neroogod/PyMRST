"""MRST ``flipGrid.m`` coordinate-axis remapping."""

from copy import deepcopy

import numpy as np


def flip_grid(g):
    """Flip a grid's coordinate axes as MRST ``flipGrid``."""
    out = deepcopy(g)
    out.nodes.coords = out.nodes.coords[:, [2, 0, 1]]
    if getattr(out.cells, "tags", None) is not None:
        tags = np.array([[0, 2, 4], [1, 3, 5]], dtype=np.int64)
        new_tag = tags[:, [1, 2, 0]].ravel(order="F")
        old = np.asarray(out.cells.tags, dtype=np.int64)
        keep = (old >= 0) & (old < 6)
        mapped = old.copy()
        mapped[keep] = new_tag[old[keep]]
        out.cells.tags = mapped
    return out

__all__ = ["flip_grid"]
