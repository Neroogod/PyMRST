"""Native port of ADFNE15 ``GetShapePoly.m``."""

import numpy as np

from .translate import translate


def get_shape_poly(*, shape="Cylinder", radius=100.0, center=(0, 0, 0),
                   length=.5, resolution=6):
    """Generate the oriented boundary polygons for an x-axis cylinder."""
    if str(shape).lower() != "cylinder":
        raise ValueError(f"unsupported shape: {shape}")
    if radius <= 0 or length <= 0 or int(resolution) < 3:
        raise ValueError("radius, length and resolution must define a cylinder")
    resolution = int(resolution)
    angles = np.linspace(0, 2*np.pi, resolution+1)
    circle = np.column_stack((np.cos(angles)*radius,
                              np.sin(angles)*radius))
    polygons = [np.column_stack((np.zeros(resolution+1), circle[::-1])),
                np.column_stack((np.full(resolution+1, length), circle))]
    for index in range(resolution):
        a, b = circle[index], circle[index+1]
        polygons.append(np.array([[0, *a], [0, *b], [length, *b],
                                  [length, *a], [0, *a]], dtype=float))
    moved = translate(polygons, center)
    return [moved] if isinstance(moved, np.ndarray) else moved
