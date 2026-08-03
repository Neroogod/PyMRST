"""Native port of ADFNE15 ``Geometry.m``."""

from dataclasses import dataclass
import numpy as np

from .rotate import rotate


@dataclass(frozen=True, slots=True)
class SphereMesh:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray


def ellipse(*, center=(0, 0), radii=(1, .5), resolution=12, angle=0):
    center = np.asarray(center, dtype=float)
    radii = np.asarray(radii, dtype=float)
    theta = np.linspace(0, 2*np.pi, int(resolution), endpoint=False)
    output = center + np.column_stack((radii[0]*np.cos(theta),
                                       radii[1]*np.sin(theta)))
    return rotate(output, angle=angle, center=center) if angle else output


def sphere(*, center=(0, 0, 0), radius=1.0, resolution=12):
    center = np.asarray(center, dtype=float)
    k = int(resolution)
    theta = np.arange(-k, k+1, 2, dtype=float)/k*np.pi
    phi = np.arange(-k, k+1, 2, dtype=float)[:, None]/k*np.pi/2
    cosine = np.cos(phi); cosine[0] = 0; cosine[-1] = 0
    sine = np.sin(theta); sine[0] = 0; sine[-1] = 0
    return SphereMesh(center[0] + cosine*np.cos(theta)*radius,
                      center[1] + cosine*sine*radius,
                      center[2] + np.sin(phi)*np.ones((1, k+1))*radius)


def ellipsoid(*, line=(0, 0, 0, 1, 1, 1), radius=.5, resolution=32,
              axis=None, **style):
    """Render the graphics-handle result returned by ADFNE Geometry/Ellipsoid."""
    from .draw import draw
    return draw("ellipsoid", line, axis=axis, radius=radius,
                resolution=resolution, **style)


def geometry(kind, **options):
    name = str(kind).lower()
    if name in {"elp", "ellipse"}:
        return ellipse(**options)
    if name in {"sph", "sphere"}:
        return sphere(**options)
    if name in {"els", "ellipsoid"}:
        return ellipsoid(**options)
    raise ValueError(f"unsupported geometry: {kind}")
