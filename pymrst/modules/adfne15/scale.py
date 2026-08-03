"""Port of numeric and centered geometry modes from ``Scale.m``."""

import numpy as np


def scale(values, input_min=None, input_max=None, output_min=0.0, output_max=1.0):
    values = np.asarray(values, dtype=float)
    if input_min is None:
        input_min, input_max = np.min(values), np.max(values)
    if input_max is None:
        output_min, output_max = float(input_min), float(input_max)
        input_min, input_max = np.min(values), np.max(values)
    if input_max == input_min:
        return np.full_like(values, .5*(output_min+output_max), dtype=float)
    return ((values-input_min)/(input_max-input_min))*(output_max-output_min)+output_min
