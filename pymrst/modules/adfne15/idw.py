"""Native port of ADFNE15 ``IDW.m``."""

import numpy as np


def inverse_distance_weighting(reference_points, reference_values, points,
                               power=2):
    reference_points = np.asarray(reference_points, dtype=float)
    reference_values = np.asarray(reference_values, dtype=float)
    points = np.atleast_2d(np.asarray(points, dtype=float))
    distances = np.linalg.norm(
        points[:, None, :]-reference_points[None, :, :], axis=2)
    exact = distances <= 1e-15
    weights = np.divide(1.0, distances**power, out=np.zeros_like(distances),
                        where=~exact)
    output = (weights @ reference_values)/np.sum(weights, axis=1)
    for row in np.flatnonzero(np.any(exact, axis=1)):
        output[row] = reference_values[np.flatnonzero(exact[row])[0]]
    return output
