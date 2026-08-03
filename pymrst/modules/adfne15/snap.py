"""Port of ADFNE15 ``Snap.m``."""

import numpy as np


def snap(points, lines, tolerance=1e-14, *, distances_only=False):
    points = np.atleast_2d(np.asarray(points, dtype=float))
    lines = np.atleast_2d(np.asarray(lines, dtype=float))
    dimension = points.shape[1]
    if lines.shape[1] != 2*dimension:
        raise ValueError("line dimension does not match points")
    start, delta = lines[:, :dimension], lines[:, dimension:]-lines[:, :dimension]
    denominator = np.sum(delta*delta, axis=1)
    relative = points[:, None, :]-start[None, :, :]
    parameter = np.divide(np.sum(relative*delta[None, :, :], axis=2), denominator,
                          out=np.zeros((len(points), len(lines))), where=denominator > np.finfo(float).eps)
    parameter = np.clip(parameter, 0, 1)
    closest = start[None, :, :]+parameter[:, :, None]*delta[None, :, :]
    distances = np.linalg.norm(points[:, None, :]-closest, axis=2)
    if distances_only or tolerance == 0:
        return distances
    mask = distances <= tolerance
    output = points.copy()
    for row in np.flatnonzero(np.any(mask, axis=1)):
        output[row] = closest[row, np.flatnonzero(mask[row])[0]]
    return output, mask
