"""Cartesian-to-active cell mapping (MRST ``cart2active``)."""

from __future__ import annotations

import numpy as np


def cart2active(grid, cart_cells, *, preserve_shape: bool = False) -> np.ndarray:
    """Map zero-based global Cartesian cells to zero-based active cells.

    By default inactive input cells are omitted, matching MRST. Set
    ``preserve_shape=True`` to retain them as ``-1`` in an array with the input
    shape.
    """
    if grid.cartDims is None:
        raise ValueError("cart2active requires grid.cartDims")
    cells = np.asarray(cart_cells, dtype=np.int64)
    flat = cells.ravel()
    total = int(np.prod(grid.cartDims))
    if np.any(flat < 0) or np.any(flat >= total):
        raise IndexError(f"Cartesian cells must be in [0, {total})")

    index_map = np.asarray(grid.cells.indexMap, dtype=np.int64).ravel()
    if np.any(index_map < 0) or np.any(index_map >= total):
        raise ValueError("cells.indexMap contains invalid Cartesian cells")
    inverse = np.full(total, -1, dtype=np.int64)
    inverse[index_map] = np.arange(index_map.size, dtype=np.int64)
    active = inverse[flat]
    if preserve_shape:
        return active.reshape(cells.shape)
    return active[active >= 0]


cart_to_active = cart2active

__all__ = ["cart2active", "cart_to_active"]
