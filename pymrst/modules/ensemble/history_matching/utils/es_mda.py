"""ES-MDA utilities (MRST history-matching layout)."""

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from pymrst.modules.ensemble.history_matching.utils.enkf import enkf_update


def esmda_alphas(n_iter: int) -> np.ndarray:
    """Standard ES-MDA inflation factors."""
    return np.full(int(n_iter), float(n_iter))


def check_alphas(alphas: Sequence[float]) -> np.ndarray:
    """Validate that ``sum(1/alpha) = 1``."""
    a = np.asarray(alphas, float).ravel()
    total = float((1.0 / a).sum())
    if not (0.999 < total < 1.001):
        raise ValueError(
            f"1/alpha must sum to 1 (the data must be assimilated exactly once); got {total:.4f}"
        )
    return a


def es_mda(
    parameters: np.ndarray,
    observations: np.ndarray,
    obs_cov: np.ndarray,
    forward: Callable[[np.ndarray], np.ndarray],
    alphas: Optional[Sequence[float]] = None,
    n_iter: int = 4,
    perturb_observations: bool = True,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, list]:
    """Ensemble Smoother with Multiple Data Assimilation."""
    rng = np.random.default_rng() if rng is None else rng
    m = np.array(np.atleast_2d(parameters), float, copy=True)
    n = m.shape[1]

    d_obs = np.asarray(observations, float).ravel()
    r = np.asarray(obs_cov, float)
    if r.ndim == 1:
        r = np.diag(r)
    a = esmda_alphas(n_iter) if alphas is None else check_alphas(alphas)

    def misfit(sim):
        res = sim - d_obs[:, None]
        w = np.linalg.solve(r, res)
        return float(np.mean(np.sum(res * w, axis=0)))

    history = []
    sim = forward(m)
    history.append(misfit(sim))

    chol = np.linalg.cholesky(r)
    for alpha in a:
        if perturb_observations:
            noise = np.sqrt(alpha) * (chol @ rng.standard_normal((r.shape[0], n)))
            target = d_obs[:, None] + noise
        else:
            target = None

        m = enkf_update(m, d_obs, sim, alpha * r, perturbed=target)
        sim = forward(m)
        history.append(misfit(sim))

    return m, history


__all__ = ["check_alphas", "es_mda", "esmda_alphas"]
