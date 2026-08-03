"""Native port of ADFNE15 ``Ray.m``."""

import numpy as np

from .center import center


def ray(polygons):
    """Create packed lines from each polygon center to every vertex."""
    collection = list(polygons) if isinstance(polygons, (list, tuple)) else [polygons]
    output = []
    for polygon in collection:
        polygon = np.asarray(polygon, dtype=float)
        centroid = np.asarray(center([polygon]), dtype=float).reshape(-1)
        output.append(np.hstack((np.repeat(centroid[None, :], len(polygon), axis=0),
                                 polygon)))
    return output[0] if len(output) == 1 else output
