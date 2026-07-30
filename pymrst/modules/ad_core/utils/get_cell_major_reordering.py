"""Variable-major to cell-major system permutation."""

from __future__ import annotations

import numpy as np


def get_cell_major_reordering(
    ncell: int,
    block_size: int,
    *,
    ndof: int | None = None,
    equation_ordering=None,
    cell_ordering=None,
) -> np.ndarray:
    """Port MRST ``getCellMajorReordering`` using zero-based indices."""
    ncell, block_size = int(ncell), int(block_size)
    total = ncell * block_size
    ndof = total if ndof is None else int(ndof)
    if ndof < total:
        raise ValueError("ndof cannot be smaller than ncell*block_size")
    blocks = np.arange(total, dtype=np.int64).reshape(block_size, ncell)
    if cell_ordering is not None:
        blocks = blocks[:, np.asarray(cell_ordering, dtype=np.int64)]
    if equation_ordering is not None:
        blocks = blocks[np.asarray(equation_ordering, dtype=np.int64), :]
    ordering = np.arange(ndof, dtype=np.int64)
    ordering[:total] = blocks.ravel(order="F")
    return ordering


getCellMajorReordering = get_cell_major_reordering

__all__ = ["get_cell_major_reordering", "getCellMajorReordering"]
