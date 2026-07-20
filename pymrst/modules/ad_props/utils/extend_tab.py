"""MRST ``extendTab.m`` flat table extension."""

import numpy as np


def extend_tab(tab: np.ndarray, pm: float = 1.0) -> np.ndarray:
    """Pad a table with flat rows at ``x[0]-pm`` and ``x[-1]+pm``."""
    tab = np.asarray(tab, dtype=float)
    first = tab[0].copy()
    first[0] -= pm
    last = tab[-1].copy()
    last[0] += pm
    return np.vstack([first, tab, last])

__all__ = ["extend_tab"]
