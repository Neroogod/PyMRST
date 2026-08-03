"""Port of ADFNE15 ``Polynorm.m``."""
import numpy as np
def polygon_normals(polygons): return np.vstack([np.cross(p[1]-p[0],p[2]-p[0]) for p in map(np.asarray,polygons)])
