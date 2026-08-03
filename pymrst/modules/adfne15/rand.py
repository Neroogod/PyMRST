"""Port of ADFNE15 ``Rand.m`` bounded distributions."""

import numpy as np

from .angle import angle


def random_values(size, *, distribution="uniform", bounds=(0.0, 1.0), mean=0.0,
                  kappa=0.0, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    shape = (int(size),) if np.isscalar(size) else tuple(size)
    name = str(distribution).lower()
    if name in {"uniform", "u"}:
        return rng.uniform(bounds[0], bounds[1], shape)
    if name in {"exp", "e", "exponential"}:
        if mean <= 0: raise ValueError("exponential mean must be positive")
        lower = 1-np.exp(-bounds[0]/mean); upper = 1-np.exp(-bounds[1]/mean)
        return -mean*np.log1p(-rng.uniform(lower, upper, shape))
    if name in {"f", "fisher", "vonmises"}:
        values = (rng.uniform(0, 360, shape) if abs(kappa) <= 1e-6 else
                  np.degrees(rng.vonmises(np.radians(mean), kappa, shape)))
        values = angle(values, 360)
        return angle(values, 90) if bounds[1] == 90 else values
    raise ValueError(f"unsupported ADFNE distribution: {distribution}")
