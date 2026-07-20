"""Uniform Cartesian partitioning (MRST ``partitionCartGrid.m`` layout)."""

from typing import Sequence

import numpy as np


def partition_cart_grid(cart_dims: Sequence[int], part_dims: Sequence[int]) -> np.ndarray:
    """Uniform Cartesian partition, returning zero-based coarse block ids."""
    cart = list(cart_dims) + [1] * (3 - len(cart_dims))
    part = list(part_dims) + [1] * (3 - len(part_dims))
    nx, ny, nz = cart
    cnx, cny, cnz = part
    i, j, k = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    bi = np.minimum((i * cnx) // nx, cnx - 1)
    bj = np.minimum((j * cny) // ny, cny - 1)
    bk = np.minimum((k * cnz) // nz, cnz - 1)
    p = bi + cnx * (bj + cny * bk)
    return np.ravel(p, order="F").astype(np.int64)


__all__ = ["partition_cart_grid"]
