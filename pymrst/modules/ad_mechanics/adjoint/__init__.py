"""MRST-shaped ad-mechanics adjoint landing modules."""

from pymrst.modules.ad_mechanics.adjoint.adjoint import (
    AdjointResult,
    compute_gradient_adjoint,
    evaluate_objective,
    simulate_forward,
    uplift_objective,
)

__all__ = [
    "AdjointResult",
    "compute_gradient_adjoint",
    "evaluate_objective",
    "simulate_forward",
    "uplift_objective",
]
