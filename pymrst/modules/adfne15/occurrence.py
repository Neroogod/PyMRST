"""Native port of ADFNE15 ``Occurrence.m``."""

import numpy as np


def occurrence(points, references, tolerance=1e-14):
    points = np.atleast_2d(np.asarray(points, dtype=float))
    references = np.asarray(references, dtype=float)
    return np.asarray([
        np.count_nonzero(np.linalg.norm(references-point, axis=-1) <= tolerance)
        for point in points], dtype=np.int64)
