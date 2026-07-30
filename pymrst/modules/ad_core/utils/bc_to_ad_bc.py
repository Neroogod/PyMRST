"""Add reduced boundary-face/cell mapping matrices to boundary conditions."""

from __future__ import annotations

from copy import copy

import numpy as np
import scipy.sparse as sp


def _get(obj, name):
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def _set(obj, name, value):
    if isinstance(obj, dict):
        obj[name] = value
    else:
        setattr(obj, name, value)


def bc_to_ad_bc(grid, boundary):
    """Port deprecated MRST ``bc2ADbc`` with zero-based face/cell indices."""
    output = copy(boundary)
    faces = np.asarray(_get(boundary, "face"), dtype=np.int64).ravel()
    neighbors = np.asarray(_get(_get(grid, "faces"), "neighbors"), dtype=np.int64)
    selected = neighbors[faces]
    native_indexing = np.any(neighbors < 0)
    exterior = selected < 0 if native_indexing else selected == 0
    if not np.all(np.sum(exterior, axis=1) == 1):
        raise ValueError("boundary conditions must be placed on boundary faces")
    cells = np.max(selected, axis=1)
    if not native_indexing:
        cells = cells - 1
    number_of_cells = int(_get(_get(grid, "cells"), "num"))
    mapping = sp.csr_matrix(
        (np.ones(faces.size), (np.arange(faces.size), cells)),
        shape=(faces.size, number_of_cells),
    )
    _set(output, "cell2bcface", mapping)
    _set(output, "bcface2cell", mapping.T.tocsr())
    return output


bc2ADbc = bc_to_ad_bc

__all__ = ["bc_to_ad_bc", "bc2ADbc"]
