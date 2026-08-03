"""Port of ADFNE15 ``Plane.m``."""
import numpy as np
from .unique import unique_rows
def plane(points,normalized=True):
    points=unique_rows(points); origin=points[0]; first=points[1]-origin; normal=np.zeros(3)
    for candidate in points[2:]:
        normal=np.cross(first,candidate-origin)
        if np.linalg.norm(normal)>1e-15: break
    if np.linalg.norm(normal)<=1e-15: raise ValueError("plane points are collinear")
    if normalized:
        first/=np.linalg.norm(first); normal/=np.linalg.norm(normal); second=-np.cross(first,normal); second/=np.linalg.norm(second)
    else: second=points[2]-origin
    return np.r_[origin,first,second]
