"""Standalone ensemble simulation (MRST ``simulateEnsembleMembersStandalone.m``)."""

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass
class EnsembleResult:
    """Quantity of interest for every member, plus statistics."""

    parameters: np.ndarray
    qoi: np.ndarray

    @property
    def mean(self) -> np.ndarray:
        return self.qoi.mean(axis=1)

    @property
    def std(self) -> np.ndarray:
        return self.qoi.std(axis=1, ddof=1)

    def percentile(self, q) -> np.ndarray:
        return np.percentile(self.qoi, q, axis=1)


def run_ensemble(parameters: np.ndarray, simulate: Callable[[np.ndarray], np.ndarray]) -> EnsembleResult:
    """Run one independent simulation per ensemble member."""
    m = np.atleast_2d(np.asarray(parameters, float))
    cols = [np.asarray(simulate(m[:, k]), float).ravel() for k in range(m.shape[1])]
    n_obs = {c.size for c in cols}
    if len(n_obs) != 1:
        raise ValueError(f"members returned different QoI lengths: {sorted(n_obs)}")
    return EnsembleResult(parameters=m, qoi=np.column_stack(cols))


__all__ = ["EnsembleResult", "run_ensemble"]
