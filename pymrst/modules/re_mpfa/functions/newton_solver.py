"""Newton-AD iteration from MRST ``re-mpfa/functions/newtonSolver.m``."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import spsolve

from pymrst.utils.adi import ADI, init_variables_adi


@dataclass
class NewtonResult:
    hydraulic_head: ADI
    previous_iteration: np.ndarray
    iterations: int
    cumulative_time: float
    elapsed_time: float
    update_norm: float

    def __iter__(self):
        # Preserve MRST's five-output unpacking convention.
        yield self.hydraulic_head
        yield self.previous_iteration
        yield self.iterations
        yield self.cumulative_time
        yield self.elapsed_time


def newton_solver(
    h_ad, h_eq, time_cum, dt, max_tol_pres_head, max_iter
):
    """Advance one step with MRST's modified-Picard/Newton-AD loop."""
    start = perf_counter()
    if not isinstance(h_ad, ADI):
        (h_ad,) = init_variables_adi(np.asarray(h_ad, dtype=float).ravel())
    else:
        h_ad = h_ad.copy()
    h_n0 = h_ad.val.copy()
    time_cum = float(time_cum) + float(dt)
    update_norm = np.inf
    previous = h_ad.val.copy()

    for iteration in range(1, int(max_iter) + 1):
        previous = h_ad.val.copy()
        equation = h_eq(h_ad, h_n0, previous, dt)
        if not isinstance(equation, ADI):
            raise TypeError("h_eq must return an ADI residual")
        update = spsolve(equation.jac[0].tocsc(), -equation.val)
        h_ad.val = h_ad.val + np.asarray(update).ravel()
        update_norm = float(np.max(np.abs(h_ad.val - previous)))
        if update_norm <= max_tol_pres_head:
            return NewtonResult(
                h_ad, previous, iteration, time_cum,
                perf_counter() - start, update_norm,
            )
    raise RuntimeError(
        "Newton method did not converge within "
        f"{max_iter} iterations (head update {update_norm:.3e})"
    )


newtonSolver = newton_solver

__all__ = ["NewtonResult", "newton_solver", "newtonSolver"]
