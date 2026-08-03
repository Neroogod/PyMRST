"""One evidence-bearing surface over existing adjoints and FD qualification."""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import Any, Callable, Mapping

import numpy as np

from pymrst.research_studio.experiments.definition import canonical_value, content_digest


def _resolve(owner):
    if callable(owner):
        return owner
    module, symbol = owner.split(":", 1)
    return getattr(importlib.import_module(module), symbol)


def ks_aggregate(normalized_margins, rho: float):
    """Conservative smooth upper bound of max(normalized_margins)."""
    values = np.asarray(normalized_margins, dtype=float)
    if values.size == 0 or rho <= 0 or np.any(~np.isfinite(values)):
        raise ValueError("KS aggregation requires finite values and positive rho")
    maximum = float(np.max(values))
    return maximum + np.log(np.sum(np.exp(rho*(values-maximum))))/rho


@dataclass(frozen=True, slots=True)
class ObjectiveDefinition:
    name: str
    owner: Any
    sense: str = "maximize"
    configuration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name or self.sense not in {"maximize", "minimize"}:
            raise ValueError("objective name and maximize/minimize sense are required")
        object.__setattr__(self, "configuration", canonical_value(self.configuration))

    def evaluate(self, variables, context):
        return float(_resolve(self.owner)(np.asarray(variables, float), context,
                                          **dict(self.configuration)))


@dataclass(frozen=True, slots=True)
class ConstraintDefinition:
    name: str
    owner: Any
    scale: float
    limit: float = 0.0
    rho: float | None = None
    configuration: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.name or self.scale <= 0 or (self.rho is not None and self.rho <= 0):
            raise ValueError("constraint name, positive scale and positive rho are required")
        object.__setattr__(self, "configuration", canonical_value(self.configuration))

    def evaluate(self, variables, context):
        raw = np.asarray(_resolve(self.owner)(np.asarray(variables, float), context,
                                              **dict(self.configuration)), float)
        normalized = (raw-self.limit)/self.scale
        return float(np.max(normalized) if self.rho is None
                     else ks_aggregate(normalized, self.rho))


@dataclass(frozen=True, slots=True)
class OptimizationProblem:
    objective: ObjectiveDefinition
    variable_names: tuple[str, ...]
    bounds: tuple[tuple[float, float], ...]
    constraints: tuple[ConstraintDefinition, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.variable_names or len(self.variable_names) != len(self.bounds):
            raise ValueError("every named variable needs bounds")
        if any(not low < high for low, high in self.bounds):
            raise ValueError("optimization bounds must increase")
        object.__setattr__(self, "metadata", canonical_value(self.metadata))

    @property
    def problem_id(self):
        owner = (self.objective.owner if isinstance(self.objective.owner, str)
                 else f"{self.objective.owner.__module__}:{self.objective.owner.__qualname__}")
        return f"optimization_{content_digest({'objective': self.objective.name, 'owner': owner, 'variables': self.variable_names, 'bounds': self.bounds, 'constraints': [c.name for c in self.constraints], 'metadata': self.metadata})}"


@dataclass(frozen=True, slots=True)
class DifferentiationResult:
    problem_id: str
    value: float
    gradient: np.ndarray
    method: str
    finite_difference_gradient: np.ndarray | None
    maximum_fd_error: float | None
    constraints: Mapping[str, float]
    variables: np.ndarray
    evidence: Mapping[str, Any]


def finite_difference_gradient(function: Callable, variables, *, relative_step=1e-6):
    x = np.asarray(variables, dtype=float)
    gradient = np.empty_like(x)
    for index in range(x.size):
        step = relative_step*max(1.0, abs(x[index]))
        plus, minus = x.copy(), x.copy()
        plus[index] += step
        minus[index] -= step
        gradient[index] = (function(plus)-function(minus))/(2.0*step)
    return gradient


def evaluate_with_gradient(problem: OptimizationProblem, variables, context, *,
                           gradient_owner=None, finite_difference_check=True,
                           relative_step=1e-6, evidence=None):
    x = np.asarray(variables, dtype=float).reshape(-1)
    if x.size != len(problem.variable_names):
        raise ValueError("variable count differs from the optimization problem")
    if any(value < low or value > high
           for value, (low, high) in zip(x, problem.bounds)):
        raise ValueError("variables lie outside optimization bounds")
    function = lambda value: problem.objective.evaluate(value, context)
    value = function(x)
    if gradient_owner is None:
        gradient = finite_difference_gradient(function, x, relative_step=relative_step)
        method = "finite-difference"
    else:
        result = _resolve(gradient_owner)(x, context)
        if isinstance(result, tuple):
            reported_value, gradient = result
            if not np.isclose(reported_value, value, rtol=1e-10, atol=1e-12):
                raise ValueError("gradient provider objective differs from declared objective")
        else:
            gradient = result
        gradient = np.asarray(gradient, dtype=float).reshape(x.shape)
        method = gradient_owner if isinstance(gradient_owner, str) else getattr(
            gradient_owner, "__name__", "callable")
    fd = (finite_difference_gradient(function, x, relative_step=relative_step)
          if finite_difference_check else None)
    maximum_error = None if fd is None else float(np.max(np.abs(gradient-fd)))
    constraints = {item.name: item.evaluate(x, context) for item in problem.constraints}
    return DifferentiationResult(
        problem.problem_id, value, gradient, str(method), fd, maximum_error,
        constraints, x.copy(), canonical_value(evidence or {}))


__all__ = ["ConstraintDefinition", "DifferentiationResult",
           "ObjectiveDefinition", "OptimizationProblem", "evaluate_with_gradient",
           "finite_difference_gradient", "ks_aggregate"]
