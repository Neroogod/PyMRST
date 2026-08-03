"""Native port of ADFNE15 ordinary ``Kriging.m``."""

from dataclasses import dataclass
import numpy as np
from scipy.spatial.distance import pdist, squareform

from .variomodel import variomodel


@dataclass(frozen=True, slots=True)
class KrigingResult:
    estimates: np.ndarray
    errors: np.ndarray
    grid_points: np.ndarray


def kriging(points, values, distances=None, model=None, grid_shape=None):
    points = np.asarray(points, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    dimension = points.shape[1]
    grid_shape = tuple([7]*dimension if grid_shape is None else grid_shape)
    condensed = pdist(points) if distances is None else np.asarray(distances, float).reshape(-1)
    gamma = squareform(variomodel(model or {"name":"exp","nugget":0,"sill":4,"range":40}, condensed))
    count = len(points)
    system = np.ones((count+1, count+1)); system[:count, :count] = gamma; system[-1, -1] = 0
    inverse = np.linalg.inv(system)
    axes = [np.linspace(np.min(points[:, i]), np.max(points[:, i]), grid_shape[i])
            for i in range(dimension)]
    mesh = np.meshgrid(*axes, indexing="xy")
    targets = np.column_stack([part.reshape(-1, order="F") for part in mesh])
    estimates = np.empty(len(targets)); errors = np.empty(len(targets))
    for index, target in enumerate(targets):
        lag = np.linalg.norm(points-target, axis=1)
        rhs = np.r_[variomodel(model or {"name":"exp","nugget":0,"sill":4,"range":40}, lag)*(lag > 0), 1]
        weights = inverse @ rhs
        estimates[index] = weights[:count] @ values
        errors[index] = weights[:count] @ rhs[:count] + weights[-1]
    shape = tuple(reversed(grid_shape))
    return KrigingResult(estimates.reshape(shape, order="F"),
                         errors.reshape(shape, order="F"), targets)
