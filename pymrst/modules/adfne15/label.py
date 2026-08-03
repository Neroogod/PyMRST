"""Native port of ADFNE15 ``Label.m``."""

import numpy as np


def label(clusters, count, *, one_based=False):
    """Assign cluster labels; isolated entries receive distinct negatives."""
    result = np.zeros(int(count), dtype=int)
    for number, members in enumerate(clusters, start=1):
        indices = np.asarray(members, dtype=int).reshape(-1)
        if one_based:
            indices = indices - 1
        result[indices] = number
    missing = result == 0
    result[missing] = -np.arange(1, np.count_nonzero(missing) + 1)
    return result
