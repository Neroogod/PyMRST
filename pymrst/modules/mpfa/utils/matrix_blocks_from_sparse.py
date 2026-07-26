"""Extract diagonal blocks like MRST ``matrixBlocksFromSparse.m``."""

import numpy as np


def matrix_blocks_from_sparse(matrix, block_sizes):
    """Return consecutive dense diagonal blocks."""
    sizes = np.asarray(block_sizes, dtype=np.int64).ravel()
    blocks = []
    offset = 0
    for size in sizes:
        blocks.append(matrix[offset:offset + size, offset:offset + size].toarray())
        offset += int(size)
    if offset != matrix.shape[0] or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("block sizes must partition a square matrix")
    return blocks


matrixBlocksFromSparse = matrix_blocks_from_sparse

__all__ = ["matrix_blocks_from_sparse", "matrixBlocksFromSparse"]
