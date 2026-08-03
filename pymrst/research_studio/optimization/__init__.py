"""Declarative optimization and gradient evidence (M4.4)."""

from .problem import (
    ConstraintDefinition, DifferentiationResult, ObjectiveDefinition,
    OptimizationProblem, evaluate_with_gradient, finite_difference_gradient,
    ks_aggregate,
)

__all__ = ["ConstraintDefinition", "DifferentiationResult",
           "ObjectiveDefinition", "OptimizationProblem",
           "evaluate_with_gradient", "finite_difference_gradient",
           "ks_aggregate"]
