"""Native port of ADFNE15 ``Wrap.m``."""

import numpy as np


def wrap(values, period):
    """Wrap one-based integer values into ``1..period`` as MATLAB does."""
    if period <= 0:
        raise ValueError("period must be positive")
    values = np.asarray(values)
    return np.mod(values - 1, period) + 1
