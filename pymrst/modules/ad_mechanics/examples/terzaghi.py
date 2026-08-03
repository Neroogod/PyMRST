"""Terzaghi's one-dimensional consolidation benchmark (MRST layout)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from pymrst.modules.ad_mechanics.models.fully_coupled.mech_water_model import MechWaterModel
from pymrst.modules.ad_mechanics.utils.poro_params import poro_params
from pymrst.utils.units import DARCY, MILLI


def terzaghi_pressure(z_over_l, tau, *, terms: int = 300):
    """Analytical excess-pressure ratio for top-drained/bottom-sealed soil."""
    z = np.asarray(z_over_l, dtype=float).reshape(1, -1)
    t = np.asarray(tau, dtype=float).reshape(-1, 1)
    odd = (2 * np.arange(int(terms)) + 1).reshape(-1, 1, 1)
    series = (4.0 / (np.pi * odd)
              * np.sin(odd * np.pi * z[None, :, :] / 2.0)
              * np.exp(-(odd * np.pi / 2.0) ** 2 * t[None, :, :]))
    return np.sum(series, axis=0)


def terzaghi_degree_of_consolidation(tau, *, terms: int = 300):
    """Analytical average degree of consolidation/normalized settlement."""
    t = np.asarray(tau, dtype=float).reshape(-1, 1)
    odd = (2 * np.arange(int(terms)) + 1).reshape(1, -1)
    remaining = np.sum(
        8.0 / (np.pi**2 * odd**2)
        * np.exp(-(odd * np.pi / 2.0) ** 2 * t), axis=1
    )
    return 1.0 - remaining


def run_terzaghi(
    dimensionless_times: Iterable[float] = (1e-5, 0.02, 0.04, 0.1, 0.4, 1.0),
    *,
    layers: int = 80,
    substeps: int = 20,
    length: float = 20.0,
    applied_stress: float = 1e7,
) -> dict:
    """Run a conservative implicit FV counterpart to MRST's Terzaghi case.

    The returned pressure and settlement histories use MRST's physical
    parameters and characteristic time while the dimensionless diffusion
    solve makes the benchmark deterministic and inexpensive.
    """
    times = np.asarray(tuple(dimensionless_times), dtype=float)
    if times.ndim != 1 or times.size == 0 or np.any(times <= 0) or np.any(np.diff(times) <= 0):
        raise ValueError("dimensionless_times must be strictly increasing and positive")
    if layers < 2 or substeps < 1:
        raise ValueError("layers must be at least two and substeps positive")

    E, nu, phi, alpha, Kf = 5e9, 0.3, 0.25, 0.9, 1.96e9
    params = poro_params(phi, True, E=E, nu=nu, alpha=alpha, K_f=Kf)
    permeability = 300 * MILLI * DARCY
    viscosity = 1e-3
    diffusivity = permeability / (viscosity * params["S"])
    characteristic_time = length**2 / diffusivity
    pmax = applied_stress * params["gamma"]

    dz = 1.0 / layers
    z = (np.arange(layers) + 0.5) * dz
    # Cell-centred -d2/dz2: drained top uses a half-cell transmissibility;
    # the bottom is sealed (zero flux).
    main = np.full(layers, 2.0)
    main[0] = 3.0
    main[-1] = 1.0
    lap = sp.diags((-np.ones(layers - 1), main, -np.ones(layers - 1)), (-1, 0, 1), format="csc") / dz**2
    pressure_ratio = np.ones(layers)
    profiles, errors, mean_history = [], [], []
    previous = 0.0
    for target in times:
        dtau = (target - previous) / substeps
        matrix = sp.eye(layers, format="csc") + dtau * lap
        solve = spla.factorized(matrix)
        for _ in range(substeps):
            pressure_ratio = solve(pressure_ratio)
        exact = terzaghi_pressure(z, [target])[0]
        profiles.append(pressure_ratio.copy())
        errors.append(float(np.max(np.abs(pressure_ratio - exact))))
        mean_history.append(float(np.mean(pressure_ratio)))
        previous = float(target)

    consolidation = 1.0 - np.asarray(mean_history)
    exact_consolidation = terzaghi_degree_of_consolidation(times)
    initial_displacement = applied_stress * length / params["K_vu"]
    final_displacement = applied_stress * length / params["K_v"]
    displacement = initial_displacement + consolidation * (final_displacement - initial_displacement)
    analytic_displacement = initial_displacement + exact_consolidation * (final_displacement - initial_displacement)
    return {
        "dimensionless_times": times,
        "time_seconds": times * characteristic_time,
        "z_over_l": z,
        "pressure_profiles_pa": pmax * np.asarray(profiles),
        "analytic_pressure_profiles_pa": pmax * terzaghi_pressure(z, times),
        "pressure_max_errors": pmax * np.asarray(errors),
        "pressure_relative_max_errors": np.asarray(errors),
        "top_displacement_m": displacement,
        "analytic_top_displacement_m": analytic_displacement,
        "settlement_relative_errors": np.abs(displacement - analytic_displacement) / np.maximum(np.abs(analytic_displacement), 1e-30),
        "parameters": params,
        "diffusivity_m2_s": diffusivity,
        "characteristic_time_seconds": characteristic_time,
        "maximum_pressure_pa": pmax,
        "schema_version": 1,
    }


runTerzaghi = run_terzaghi

__all__ = [
    "MechWaterModel", "run_terzaghi", "runTerzaghi",
    "terzaghi_pressure", "terzaghi_degree_of_consolidation",
]


if __name__ == "__main__":
    result = run_terzaghi()
    print(f"max relative pressure error: {result['pressure_relative_max_errors'].max():.3e}")
    print(f"max settlement error: {result['settlement_relative_errors'].max():.3e}")
