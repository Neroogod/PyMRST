"""Native port of ADFNE15 ``Variocloud.m``."""

from dataclasses import dataclass
import numpy as np

from .flatri import flatri


@dataclass(frozen=True, slots=True)
class VariogramCloud:
    distances: np.ndarray
    semivariances: np.ndarray
    variance: float


def variocloud(points, values, distances=None):
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(points) != len(values):
        raise ValueError("one value is required for every point")
    if distances is None:
        differences = points[:, None, :]-points[None, :, :]
        distances = np.linalg.norm(differences, axis=2)
    distances = np.asarray(distances, dtype=float)
    gamma = .5*(values[:, None]-values[None, :])**2
    return VariogramCloud(flatri(distances), flatri(gamma),
                          float(np.var(values, ddof=1)))
