"""Native port of ADFNE15 ``Relabel.m``."""

import numpy as np


def relabel(labels):
    """Relabel positive clusters from least to most frequent."""
    labels = np.asarray(labels, dtype=int)
    result = labels.copy()
    highest = int(labels.max(initial=0))
    if highest == 0:
        return result
    frequencies = np.array([np.count_nonzero(labels == i)
                            for i in range(1, highest + 1)])
    # MATLAB sort is stable, so equal-frequency labels retain label order.
    order = np.argsort(frequencies, kind="stable") + 1
    for new, old in enumerate(order, start=1):
        result[labels == old] = new
    return result
