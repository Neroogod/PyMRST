"""Port of MRST ``getUpdatesFromPressure.m`` using native PyMRST flash."""

from __future__ import annotations

import numpy as np


def get_updates_from_pressure(model, state, dp):
    """Return natural-variable updates induced by a pressure increment.

    MRST computes these updates from an AD linearization of the flash equations.
    PyMRST's compatibility implementation evaluates the same state response by
    perturbing pressure, refreshing the flash, and differencing phase variables.
    """
    pure_liquid, pure_vapour, two_phase = model.getFlag(state)
    del pure_liquid, pure_vapour
    ncomp = state.z.shape[1]
    nph = 3 if getattr(model, "water", False) else 2
    if not np.any(two_phase):
        return (
            np.zeros((0, ncomp)),
            np.zeros((0, ncomp)),
            np.zeros((0, nph)),
            np.array([]),
            two_phase,
            np.array([]),
        )

    trial = state.copy()
    dp = np.asarray(dp, dtype=float).reshape(-1)
    trial.pressure = np.asarray(trial.pressure, dtype=float).copy()
    trial.pressure[two_phase] = trial.pressure[two_phase] + dp[two_phase]
    trial = model._sync_phase_state_from_flash(trial)

    dx = trial.x[two_phase, :] - state.x[two_phase, :]
    dy = trial.y[two_phase, :] - state.y[two_phase, :]
    if getattr(model, "water", False):
        ds = np.column_stack([
            trial.sW[two_phase] - state.sW[two_phase],
            trial.sO[two_phase] - state.sO[two_phase],
            trial.sG[two_phase] - state.sG[two_phase],
        ])
    else:
        ds = np.column_stack([
            trial.sO[two_phase] - state.sO[two_phase],
            trial.sG[two_phase] - state.sG[two_phase],
        ])
    dL = trial.L[two_phase] - state.L[two_phase]
    tol = 0.1 * float(getattr(model, "nonlinearTolerance", 1.0e-3))
    with np.errstate(divide="ignore", invalid="ignore"):
        wx = np.abs(trial.x[two_phase, :] - state.x[two_phase, :]) / np.maximum(np.abs(dx), 1.0e-300)
        wy = np.abs(trial.y[two_phase, :] - state.y[two_phase, :]) / np.maximum(np.abs(dy), 1.0e-300)
    wx[np.abs(dx) < tol] = 1.0
    wy[np.abs(dy) < tol] = 1.0
    w = np.minimum(wx, wy).min(axis=1)
    w = np.clip(w, 0.0, 1.0)
    dx = dx * w[:, None]
    dy = dy * w[:, None]
    ds = ds * w[:, None]
    dL = dL * w
    return dx, dy, ds, dL, two_phase, w


def getUpdatesFromPressure(model, state, dp):
    """MRST camelCase alias for :func:`get_updates_from_pressure`."""
    return get_updates_from_pressure(model, state, dp)


__all__ = ["get_updates_from_pressure", "getUpdatesFromPressure"]
