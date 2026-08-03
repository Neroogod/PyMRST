"""Use Victory's prediction as the starting guess for PyMRST's flash.

The other option in this package, `victory_backend`, lets the network *produce*
the phase split - fast, but its residuals sit around 6e-3 where PyMRST's are
2e-11, which is too loose for a Newton loop that measures convergence with
exactly those residuals.

This is the other way round, and it has no accuracy cost at all. Peng--Robinson
still does the solving and still produces the answer; the network only supplies
where to *start*. Successive substitution converges to the same fixed point from
any reasonable initial K, so a better starting guess buys iterations, not
error - and a 6e-3-accurate surrogate is an excellent starting guess even
though it is a poor final answer.

**PyMRST is untouched.** `eos.flash` is not modified, wrapped or patched; this
runs the same iteration over the same public primitives - `fugacity_coefficients`
and `rachford_rice` - seeded differently. A caller who does not import this
module sees nothing.

**The iteration mirrors `eos.flash`'s and must be kept in step with it.** That
is the real cost of not modifying PyMRST, and it is guarded rather than trusted:
`test_it_agrees_with_pymrst_to_solver_tolerance` compares this against
`eos.flash` on every row, so if PyMRST's loop changes and this one does not, the
test fails rather than the two drifting quietly apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np

from pymrst.modules.compositional.eos import (
    FlashResult,
    fugacity_coefficients,
    rachford_rice,
    wilson_k,
)

TEMPERATURE_K = 353.0


@dataclass
class SeededFlash:
    """A flash result plus what the seed bought."""

    result: FlashResult
    iterations: int
    baseline_iterations: Optional[int] = None

    @property
    def iterations_saved(self) -> Optional[int]:
        if self.baseline_iterations is None:
            return None
        return self.baseline_iterations - self.iterations


def victory_seed_k(ensemble, p, z, two_phase, temperature=TEMPERATURE_K,
                   strict_domain: bool = True) -> np.ndarray:
    """Equilibrium ratios implied by Victory's predicted split."""
    p = np.asarray(p, float).ravel()
    z = np.atleast_2d(np.asarray(z, float))
    K = np.ones_like(z)
    if not np.any(two_phase):
        return K

    flags = np.zeros((int(two_phase.sum()), 2), dtype=np.float32)
    flags[:, 1] = 1.0                                   # [Gas, 2phase]
    pred = ensemble.predict(p[two_phase], z[two_phase], phase_flags=flags,
                            temperature=temperature,
                            strict_domain=strict_domain)
    x = np.maximum(np.asarray(pred.x, float), 1e-12)
    y = np.maximum(np.asarray(pred.y, float), 1e-12)
    K[two_phase] = y / x
    return K


def flash_with_victory_seed(mix, p, T, z, ensemble, *, tol: float = 1e-10,
                            max_it: int = 200, temperature=TEMPERATURE_K,
                            strict_domain: bool = True,
                            count_baseline: bool = False) -> SeededFlash:
    """Peng--Robinson flash, started from Victory's predicted split."""
    from pymrst.modules.compositional.utils.phase_stability_test import (
        phaseStabilityTest,
    )

    p = np.atleast_1d(np.asarray(p, float)).ravel()
    T = np.broadcast_to(np.atleast_1d(np.asarray(T, float)).ravel(), p.shape)
    z = np.atleast_2d(np.asarray(z, float))
    z = np.maximum(z, 1e-8)
    z = z / z.sum(axis=1, keepdims=True)

    # Stability is PyMRST's, exactly as in eos.flash. Victory cannot do it -
    # it takes the phase flags as inputs - and it is not asked to.
    stable, _, _ = phaseStabilityTest(mix, z, p, T, K=wilson_k(mix, p, T))
    stable = np.asarray(stable, dtype=bool)
    active = ~stable

    # The one difference from eos.flash: where the iteration starts.
    K = wilson_k(mix, p, T)
    seeded = victory_seed_k(ensemble, p, z, active, temperature, strict_domain)
    K[active] = seeded[active]

    weighted = z * np.asarray(mix.Vcrit, float)[None, :]
    critical_T = np.sum(weighted * np.asarray(mix.Tcrit, float)[None, :],
                        axis=1) / np.sum(weighted, axis=1)
    V = np.zeros(p.size)
    V[stable & ~(T < critical_T)] = 1.0
    if np.any(active):
        V[active] = rachford_rice(z[active], K[active])

    result, iterations = _ssi(mix, p, T, z, K, V, active.copy(), stable,
                              tol, max_it)

    baseline = None
    if count_baseline:
        from pymrst.modules.compositional.eos import flash
        baseline = flash(mix, p, T, z, tol=tol, max_it=max_it).iterations
    return SeededFlash(result=result, iterations=iterations,
                       baseline_iterations=baseline)


def _ssi(mix, p, T, z, K, V, active, stable, tol, max_it
         ) -> Tuple[FlashResult, int]:
    """Successive substitution - the same loop `eos.flash` runs.

    Kept here rather than imported because `eos.flash` takes no initial-K
    argument and PyMRST is deliberately not modified. The agreement test is
    what stops the two versions drifting.
    """
    it = 0
    for it in range(1, max_it + 1):
        if not np.any(active):
            it -= 1
            break
        idx = np.flatnonzero(active)
        k_local, v_local = K[idx], V[idx]
        x_local = z[idx] / (1.0 + v_local[:, None] * (k_local - 1.0))
        y_local = k_local * x_local
        x_local /= x_local.sum(axis=1, keepdims=True)
        y_local /= y_local.sum(axis=1, keepdims=True)
        ln_phi_l, _ = fugacity_coefficients(mix, p[idx], T[idx], x_local,
                                            vapour=False)
        ln_phi_v, _ = fugacity_coefficients(mix, p[idx], T[idx], y_local,
                                            vapour=True)
        k_new = np.exp(ln_phi_l - ln_phi_v)
        row_error = np.max(np.abs(k_new / k_local - 1.0), axis=1)
        K[idx] = k_new
        V[idx] = rachford_rice(z[idx], k_new)
        boundary = (V[idx] <= 1e-12) | (V[idx] >= 1.0 - 1e-12)
        active[idx[(row_error <= tol) | boundary]] = False

    x = z / (1.0 + V[:, None] * (K - 1.0))
    y = K * x
    x /= x.sum(axis=1, keepdims=True)
    y /= y.sum(axis=1, keepdims=True)
    x[stable] = z[stable]
    y[stable] = z[stable]
    _, Z_L = fugacity_coefficients(mix, p, T, x, vapour=False)
    _, Z_V = fugacity_coefficients(mix, p, T, y, vapour=True)
    return FlashResult(V=V, x=x, y=y, K=K, Z_L=Z_L, Z_V=Z_V,
                       two_phase=(V > 1e-8) & (V < 1.0 - 1e-8),
                       iterations=it), it


__all__ = ["SeededFlash", "flash_with_victory_seed", "victory_seed_k"]
