"""Recover mechanical controls from stress data (MRST `simple_matching_example`).

An inverse problem, and a self-consistency check on the whole AD calibration
stack. A known control vector is used to generate a stress field; that field
is handed back as "observed data"; and the calibration is asked to recover the
controls it came from.

The controls set the boundary strain and orientation, per-layer stiffness and
density, and the load. The objective compares four derived quantities -
vertical stress, maximum and minimum principal stress, and principal
orientation - so a match means the recovered model reproduces the stress
state, which is what a real calibration is asked to do.

Recovery is not expected to be exact, and measurably is not. The objective
falls by more than an order of magnitude and the principal *orientation* is
recovered closely, but the principal stress magnitudes are not: several
control vectors give a comparable objective, so the problem is under-determined
by this data. MRST's own script prints the differences rather than asserting
they are small, and reproducing that honestly is the point.

Using MRST's optimizer settings rather than the calibration defaults changes
the final objective by under a percent, so the plateau is a property of the
inverse problem, not of the tuning.
"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Dict, List, Sequence

import numpy as np

from pymrst.gridprocessing import cart_grid, compute_geometry
from pymrst.modules.vemmech.ad_version.calibration.utils.sigma2smax_smin_theta import (
    sigma_to_smax_smin_theta,
)
from pymrst.modules.vemmech.ad_version.optimize_mech import calibrate_model
from pymrst.modules.vemmech.utils.calculate_stress_vem import calculate_stress_vem
from pymrst.modules.vemmech.vem import create_augmented_grid

KILO = 1000.0

# MRST's target control vector: boundary, stiffness, Poisson ratio, load
U_TARGET = np.array([1 / 6, 0.4, 0.6,
                     0.2, 0.5, 0.3,
                     0.7, 0.1, 0.2,
                     0.2, 0.5, 0.4])


@dataclass
class MatchingResult:
    grid: object
    u_target: np.ndarray
    u_opt: np.ndarray
    target: Dict[str, np.ndarray]
    matched: Dict[str, np.ndarray]
    mean_error: Dict[str, float] = field(default_factory=dict)
    initial_error: Dict[str, float] = field(default_factory=dict)
    objective: float = float("nan")
    initial_objective: float = float("nan")


def _stress_quantities(grid, uu, extra):
    """The four quantities the objective is built from.

    MRST composes the stress operator from the solve's own pieces,
    ``D * WC' * assemb'``, rather than rebuilding it from the tensor, so the
    stress is consistent with the displacement that produced it.
    """
    stress_op = extra.D @ extra.WC.T @ extra.assemb.T
    stress, _ = calculate_stress_vem(grid, uu, stress_op=stress_op)
    stress = np.asarray(stress)
    smax, smin, theta = sigma_to_smax_smin_theta(
        stress[:, 0], stress[:, 1], stress[:, 3], gdim=3)
    return {"Sv": stress[:, 2], "Smax": np.asarray(smax),
            "Smin": np.asarray(smin), "theta": np.asarray(theta)}


def run_simple_matching_example(
    *,
    grid_res: Sequence[int] = (10, 10, 10),
    grid_dims: Sequence[float] = (3 * KILO, 3 * KILO, 1 * KILO),
    young: float = 5e9,
    poisson: float = 0.25,
    rho: float = 1500.0,
    layer_start: Sequence[int] = (1, 4, 7),
    change_rate: float = 1.0,
    max_it: int = 30,
    u_target: Sequence[float] = None,
) -> MatchingResult:
    """Generate synthetic stress data, then calibrate the model back to it."""
    u_target = U_TARGET if u_target is None else np.asarray(u_target, dtype=float)
    grid = create_augmented_grid(
        compute_geometry(cart_grid(list(grid_res), list(grid_dims))))
    nc = grid.cells.num
    props = SimpleNamespace(young=np.full(nc, young),
                            poisson=np.full(nc, poisson),
                            rho=np.full(nc, rho))

    # mean and rate of change with the control parameter, per MRST
    bcond = np.array([[0.0, np.pi / 4],        # orientation
                      [0.0, 1e-4],             # first strain
                      [-2e-4, 1e-4]])          # second strain, breaks isotropy
    layers = np.asarray(layer_start, dtype=float).reshape(-1, 1)
    matcond = np.hstack([layers, np.ones((layers.shape[0], 3)) * change_rate])

    synthetic = calibrate_model(grid, props, bcond, matcond, None,
                                produce_syndata_for=u_target)
    target = _stress_quantities(grid, synthetic.uu, synthetic.extra)

    cells = np.arange(nc)
    weight = np.ones(nc)
    data = {name: np.column_stack([cells, target[name], weight])
            for name in ("Sv", "orientation", "Smax", "Smin")
            if name != "orientation"}
    data["orientation"] = np.column_stack([cells, target["theta"], weight])

    # optimizer settings transcribed from MRST's optimizeSR1 call
    result = calibrate_model(grid, props, bcond, matcond, data,
                             ignore_prior=True, max_it=max_it,
                             delta=0.1, b_scale=0.01, eta=1e-3,
                             grad_tol=1e-16, obj_change_tol=1e-16)
    # the calibration returns its optimum and the solve that produced it, so
    # the matched stresses come from the optimizer rather than a re-run
    u_opt = np.asarray(result.u_opt, dtype=float)
    matched = _stress_quantities(grid, result.uu_opt, result.extra)
    initial = _stress_quantities(grid, result.uu_init, result.extra_init)

    # per-cell RMS, not a mean: `theta` is an angle, so its mean says little,
    # and a mean can look good while every cell is wrong
    def rms(a, b):
        scale = max(float(np.sqrt(np.mean(np.asarray(a) ** 2))), 1e-30)
        return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2))) / scale

    errors = {name: rms(target[name], matched[name]) for name in target}
    initial_errors = {name: rms(target[name], initial[name]) for name in target}
    return MatchingResult(grid=grid, u_target=u_target, u_opt=u_opt,
                          target=target, matched=matched, mean_error=errors,
                          initial_error=initial_errors,
                          objective=float(result.foptval),
                          initial_objective=float(result.obj_fun(result.u_init)[0]
                                                  if callable(result.obj_fun)
                                                  else np.nan))


if __name__ == "__main__":
    result = run_simple_matching_example()
    print(f"calibration on {result.grid.cells.num} cells, "
          f"{result.u_target.size} controls")
    print(f"  objective  {result.initial_objective:.6e} -> {result.objective:.6e}")
    print(f"  max control change: "
          f"{np.abs(result.u_opt - result.u_target).max():.4f}")
    print("  relative RMS against the target stresses:")
    for name in ("theta", "Smax", "Smin", "Sv"):
        print(f"    {name:6}: {result.initial_error[name]:.3e} -> "
              f"{result.mean_error[name]:.3e}")
    assert all(np.all(np.isfinite(v)) for v in result.matched.values())
    print("\nExample ran.")
