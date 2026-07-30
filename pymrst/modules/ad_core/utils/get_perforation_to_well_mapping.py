"""Map flattened well connections to their owning wells."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def get_perforation_to_well_mapping(wells):
    """Port MRST ``getPerforationToWellMapping`` using zero-based well indices."""
    wells = list(wells or [])
    if not wells:
        return np.zeros(0, dtype=np.int64), sp.csr_matrix((0, 0))
    counts = np.array([np.asarray(well.cells).size for well in wells], dtype=np.int64)
    mapping = np.repeat(np.arange(len(wells), dtype=np.int64), counts)
    if mapping.size == len(wells):
        reduction = 1.0
    else:
        reduction = sp.csr_matrix(
            (np.ones(mapping.size), (np.arange(mapping.size), mapping)),
            shape=(mapping.size, len(wells)),
        )
    return mapping, reduction


getPerforationToWellMapping = get_perforation_to_well_mapping

__all__ = ["get_perforation_to_well_mapping", "getPerforationToWellMapping"]
