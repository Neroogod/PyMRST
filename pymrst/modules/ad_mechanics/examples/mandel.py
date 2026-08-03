"""Runnable MRST-layout Mandel poroelasticity example."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from pymrst.modules.ad_mechanics.mandel import (
    MandelCase,
    consolidation_coefficient,
    mandel_pressure,
    mandel_roots,
    poro_params,
)


def run_mandel(
    dimensionless_times: Iterable[float] = (0.01, 0.1, 0.5, 1.0, 2.0),
    *,
    nx: int = 40,
    nz: int = 6,
    substeps: int = 10,
) -> dict:
    """Run Mandel's constant-load problem and return simulation/error data."""
    times = np.asarray(tuple(dimensionless_times), dtype=float)
    if times.ndim != 1 or times.size == 0 or np.any(times <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("dimensionless_times must be a strictly increasing positive sequence")
    if substeps < 1:
        raise ValueError("substeps must be positive")

    case = MandelCase(nx=nx, nz=nz)
    state, plate = case.initial_state()
    x = case.G.cells.centroids[:, 0] / case.a
    p0 = float(state["pressure"][np.argmin(x)])
    states, profiles, errors = [], [], []
    previous = 0.0
    for tau in times:
        dt = (tau - previous) * case.a**2 / case.cv / substeps
        for _ in range(substeps):
            state, plate = case.solve_step(state, dt, plate)
        simulated = np.asarray(state["pressure"], dtype=float) / p0
        analytic = mandel_pressure(x, np.array([tau]), case.nu, case.params["nu_u"])[0]
        states.append(state.copy())
        profiles.append(simulated)
        errors.append(float(np.max(np.abs(simulated - analytic))))
        previous = float(tau)
    return {
        "case": case,
        "states": states,
        "dimensionless_times": times,
        "x_over_a": x,
        "pressure_profiles": np.asarray(profiles),
        "analytic_profiles": mandel_pressure(x, times, case.nu, case.params["nu_u"]),
        "max_errors": np.asarray(errors),
        "plate_displacement": plate,
        "schema_version": 1,
        "pressure_normalization_pa": p0,
    }


runMandel = run_mandel

__all__ = [
    "MandelCase", "consolidation_coefficient", "mandel_pressure",
    "mandel_roots", "poro_params", "run_mandel", "runMandel",
]
