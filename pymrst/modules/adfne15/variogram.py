"""Native port of ADFNE15 ``Variogram.m``."""

from dataclasses import dataclass
import numpy as np

from .variocloud import variocloud


@dataclass(frozen=True, slots=True)
class ExperimentalVariogram:
    lags: np.ndarray
    values: np.ndarray
    counts: np.ndarray
    variance: float


def variogram(distances, semivariances, variance=0.0, minimum_lag_factor=3):
    distances = np.asarray(distances, dtype=float)
    if distances.ndim == 2 and distances.shape[1] > 1:
        cloud = variocloud(distances, semivariances)
        distances, semivariances, variance = (
            cloud.distances, cloud.semivariances, cloud.variance)
    distances = distances.reshape(-1)
    semivariances = np.asarray(semivariances, dtype=float).reshape(-1)
    if len(distances) != len(semivariances) or not len(distances):
        raise ValueError("distance and semivariance vectors must be nonempty and aligned")
    lag = minimum_lag_factor*np.min(distances)
    if lag <= 0:
        raise ValueError("pair distances must be positive")
    count = int(np.floor((.5*np.max(distances))/lag))
    mapped = np.ceil(distances/lag).astype(int)
    lags = np.empty(count); values = np.empty(count); populations = np.empty(count, int)
    for index in range(1, count+1):
        selected = mapped == index
        lags[index-1] = np.mean(distances[selected]) if np.any(selected) else np.nan
        values[index-1] = np.mean(semivariances[selected]) if np.any(selected) else np.nan
        populations[index-1] = np.count_nonzero(selected)
    return ExperimentalVariogram(lags, values, populations, float(variance))
