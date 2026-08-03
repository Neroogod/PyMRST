"""Port of ADFNE15 ``Angle.m``."""

import numpy as np


def angle(values, limit=180):
    values = np.asarray(values, dtype=float)
    if limit == 0:
        return np.mod(90-values, 360)
    if limit == 90:
        output = np.mod(values, 360)
        output = np.mod(output, 180)
        return np.where(output > 90, 180-output, output)
    if limit == 360:
        return np.mod(values, 360)
    output = values.copy()
    mask = (output < 0) | (output > 180)
    output[mask] = np.abs(-(-np.abs(output[mask])+180))
    return np.where(output > 90, 180-output, output)
