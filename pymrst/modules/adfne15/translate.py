"""Native port of ADFNE15 ``Translate.m``."""

import numpy as np


def translate(values, movement):
    """Translate points, packed lines, or a collection of polygons."""
    movement = np.asarray(movement, dtype=float).reshape(-1)
    if isinstance(values, (list, tuple)):
        moved = [np.asarray(value, dtype=float) + movement for value in values]
        return moved[0] if len(moved) == 1 else moved
    values = np.asarray(values, dtype=float)
    offset = movement if values.shape[-1] == movement.size else np.tile(
        movement, values.shape[-1] // movement.size)
    if offset.size != values.shape[-1]:
        raise ValueError("movement dimension is incompatible with input")
    return values + offset
