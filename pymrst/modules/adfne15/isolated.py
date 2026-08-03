"""Port of ADFNE15 ``Isolated.m``."""

import numpy as np

from .occurrence import occurrence


def isolated(lines, tolerance=1e-14):
    lines = np.asarray(lines, dtype=float); dimension = lines.shape[1]//2
    endpoints = np.vstack((lines[:, :dimension], lines[:, dimension:]))
    return np.asarray([(occurrence(line[:dimension], endpoints, tolerance)[0] <= 1)
                       or (occurrence(line[dimension:], endpoints, tolerance)[0] <= 1)
                       for line in lines], dtype=bool)
