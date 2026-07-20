"""Ensemble Kalman analysis step (MRST ``enkf`` layout)."""

from typing import Optional

import numpy as np


def enkf_update(
    parameters: np.ndarray,
    observations: np.ndarray,
    simulated: np.ndarray,
    obs_cov: np.ndarray,
    perturbed: Optional[np.ndarray] = None,
) -> np.ndarray:
    """One ensemble Kalman analysis step."""
    m = np.atleast_2d(np.asarray(parameters, float))
    d = np.atleast_2d(np.asarray(simulated, float))
    n_obs, n = d.shape
    assert m.shape[1] == n, "parameters and simulations must share the ensemble"
    assert n > 1, "an ensemble of one has no covariance to work with"

    r = np.atleast_2d(np.asarray(obs_cov, float))
    target = (
        np.asarray(observations, float).reshape(-1, 1)
        if perturbed is None
        else np.asarray(perturbed, float)
    )

    s = (d - d.mean(axis=1, keepdims=True)) / np.sqrt(n - 1)
    if n_obs <= n:
        c = s @ s.T + r
        w = np.linalg.solve(c.T, s).T
    else:
        rinv_s = np.linalg.solve(r, s)
        a = np.eye(n) + s.T @ rinv_s
        w = np.linalg.solve(a, rinv_s.T)

    anomaly = (m - m.mean(axis=1, keepdims=True)) / np.sqrt(n - 1)
    return m + anomaly @ (w @ (target - d))


__all__ = ["enkf_update"]
