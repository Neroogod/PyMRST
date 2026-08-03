"""Independent finite-difference qualification of PyMRST AD derivatives.

The finite-difference side calls the numerical function from values only.  It
therefore does not reuse AD operator implementations.  Variables are perturbed
in dimensionless coordinates so pressure, saturation and well-rate columns can
be checked together without one unit system dominating the step size.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass(frozen=True, slots=True)
class DirectionalDerivativeResult:
    directions: int
    maximum_relative_l2_error: float
    maximum_scaled_component_error: float
    maximum_fd_refinement_error: float
    maximum_absolute_l2_error: float
    passed: bool

    def to_dict(self) -> dict[str, float | int | bool]:
        return {
            "directions": self.directions,
            "maximum_relative_l2_error": self.maximum_relative_l2_error,
            "maximum_scaled_component_error": self.maximum_scaled_component_error,
            "maximum_fd_refinement_error": self.maximum_fd_refinement_error,
            "maximum_absolute_l2_error": self.maximum_absolute_l2_error,
            "passed": self.passed,
        }


def deterministic_directions(size: int, *, random_count: int = 8,
                             seed: int = 20240803) -> np.ndarray:
    """Return coordinate probes plus reproducible dense Rademacher probes."""
    if size < 1:
        raise ValueError("derivative domain must be nonempty")
    coordinate = np.eye(size) if size <= 16 else np.eye(size)[:8]
    rng = np.random.default_rng(seed)
    dense = rng.choice((-1.0, 1.0), size=(int(random_count), size))
    dense /= np.linalg.norm(dense, axis=1)[:, None]
    return np.vstack((coordinate, dense))


def qualify_directional_derivative(
    jacobian,
    numerical: Callable[[np.ndarray], np.ndarray],
    point: Sequence[float],
    *,
    scales: Sequence[float] | None = None,
    directions: np.ndarray | None = None,
    step: float = 2.0e-5,
    relative_tolerance: float = 2.0e-5,
    component_tolerance: float = 2.0e-4,
    absolute_tolerance: float = 1.0e-12,
) -> DirectionalDerivativeResult:
    """Compare ``J @ direction`` with Richardson-refined central differences.

    ``jacobian`` differentiates with respect to the physical variables.
    ``scales`` maps dimensionless perturbations to those variables.  The
    reported refinement error is the difference between central differences
    at ``h`` and ``h/2`` and exposes an unreliable finite-difference oracle.
    """
    x = np.asarray(point, dtype=float).ravel()
    scale = np.ones_like(x) if scales is None else np.asarray(scales, dtype=float).ravel()
    if scale.shape != x.shape or np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("scales must be finite, positive, and match point")
    probes = deterministic_directions(x.size) if directions is None else np.asarray(directions, float)
    if probes.ndim != 2 or probes.shape[1] != x.size:
        raise ValueError("directions must have one column per variable")
    j_scaled = jacobian @ np.diag(scale)
    max_l2 = max_component = max_refinement = max_absolute = 0.0
    for direction in probes:
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm == 0:
            raise ValueError("directions must be finite and nonzero")
        d = direction / norm

        def central(h):
            delta = scale * (h * d)
            plus = np.asarray(numerical(x + delta), dtype=float).ravel()
            minus = np.asarray(numerical(x - delta), dtype=float).ravel()
            if plus.shape != minus.shape:
                raise ValueError("numerical derivative changed output shape")
            return (plus - minus) / (2.0 * h)

        coarse = central(step)
        fine = central(step / 2.0)
        # O(h^2) central difference; Richardson removes its leading term.
        finite = (4.0 * fine - coarse) / 3.0
        automatic = np.asarray(j_scaled @ d, dtype=float).ravel()
        if automatic.shape != finite.shape:
            raise ValueError("Jacobian and numerical function output sizes differ")
        absolute = float(np.linalg.norm(automatic - finite))
        floor = max(np.linalg.norm(finite), np.linalg.norm(automatic),
                    float(absolute_tolerance))
        max_absolute = max(max_absolute, absolute)
        max_l2 = max(max_l2, absolute / floor)
        component_floor = float(absolute_tolerance) / np.sqrt(max(finite.size, 1))
        denominator = np.maximum.reduce((np.abs(automatic), np.abs(finite),
                                         np.full(finite.shape, component_floor)))
        max_component = max(max_component,
                            float(np.max(np.abs(automatic - finite) / denominator)))
        max_refinement = max(max_refinement,
                             float(np.linalg.norm(fine - coarse) / floor))
    passed = bool(
        (max_l2 <= relative_tolerance and
         max_component <= component_tolerance)
        or max_absolute <= absolute_tolerance
    )
    return DirectionalDerivativeResult(
        directions=int(probes.shape[0]),
        maximum_relative_l2_error=max_l2,
        maximum_scaled_component_error=max_component,
        maximum_fd_refinement_error=max_refinement,
        maximum_absolute_l2_error=max_absolute,
        passed=passed,
    )


__all__ = ["DirectionalDerivativeResult", "deterministic_directions",
           "qualify_directional_derivative"]
