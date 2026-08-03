"""Port of ADFNE15 ``Orientation.m``."""

from dataclasses import dataclass
import numpy as np

from .angle import angle
from .plane import plane


@dataclass(frozen=True, slots=True)
class Orientation:
    direction: np.ndarray
    dip: np.ndarray | None = None


def orientation(items):
    if isinstance(items, (list, tuple)):
        normals = np.vstack([np.cross(plane(p)[3:6], plane(p)[6:9]) for p in items])
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        direction = angle(np.degrees(np.arctan2(normals[:, 1], normals[:, 0])), 360)
        dip = angle(np.degrees(np.pi/2-np.arcsin(np.clip(normals[:, 2], -1, 1))), 90)
        return Orientation(direction, dip)
    lines = np.atleast_2d(np.asarray(items, dtype=float)); dimension = lines.shape[1]//2
    delta = lines[:, dimension:]-lines[:, :dimension]
    direction = angle(np.degrees(np.arctan2(delta[:, 1], delta[:, 0])), 360)
    if dimension == 2:
        return Orientation(direction)
    rising = delta[:, 2] > 0
    direction = np.where(rising, angle(direction+180, 360), direction)
    horizontal = np.linalg.norm(delta[:, :2], axis=1)
    dip = np.degrees(np.arctan2(np.abs(delta[:, 2]), horizontal))
    return Orientation(direction, dip)
