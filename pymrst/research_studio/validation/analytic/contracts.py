"""Registry contracts for analytic and manufactured-solution benchmarks.

This module is intentionally free of benchmark imports so the registry
contract can be used without loading any numerical owner.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

import numpy as np


@dataclass(frozen=True, slots=True)
class AnalyticCaseSpec:
    case_id: str
    name: str
    assumptions: tuple[str, ...]
    domain: str
    variables: tuple[str, ...]
    units: Mapping[str, str]
    expected_order: float | None
    minimum_order: float | None = None

    def __post_init__(self):
        if not self.case_id or not self.name or not self.assumptions:
            raise ValueError("analytic cases require identity, name, and assumptions")
        if self.expected_order is not None and self.expected_order <= 0:
            raise ValueError("expected convergence order must be positive")
        object.__setattr__(self, "units", dict(self.units))

    def to_dict(self):
        return {
            "case_id": self.case_id, "name": self.name,
            "assumptions": list(self.assumptions), "domain": self.domain,
            "variables": list(self.variables), "units": dict(self.units),
            "expected_order": self.expected_order,
            "minimum_order": self.minimum_order,
        }


@dataclass(frozen=True, slots=True)
class RefinementLevel:
    resolution: int
    characteristic_size: float
    timestep: float | None
    steps: int | None
    errors: Mapping[str, float]
    invariants: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.resolution < 1 or self.characteristic_size <= 0:
            raise ValueError("invalid refinement level")
        object.__setattr__(self, "errors", {str(k): float(v) for k, v in self.errors.items()})
        object.__setattr__(self, "invariants", {str(k): float(v) for k, v in self.invariants.items()})

    def to_dict(self):
        return {
            "resolution": self.resolution,
            "characteristic_size": self.characteristic_size,
            "timestep": self.timestep, "steps": self.steps,
            "errors": dict(self.errors), "invariants": dict(self.invariants),
        }


@dataclass(frozen=True, slots=True)
class AnalyticCaseResult:
    spec: AnalyticCaseSpec
    levels: tuple[RefinementLevel, ...]
    observed_orders: Mapping[str, float]
    passed: bool
    acceptance: Mapping[str, float]

    def to_dict(self):
        return {
            "spec": self.spec.to_dict(),
            "levels": [level.to_dict() for level in self.levels],
            "observed_orders": dict(self.observed_orders),
            "passed": self.passed, "acceptance": dict(self.acceptance),
        }


def observed_order(levels: tuple[RefinementLevel, ...], norm: str) -> float:
    """Least-squares slope of log(error) against log(characteristic size)."""
    h = np.asarray([level.characteristic_size for level in levels], float)
    error = np.asarray([level.errors[norm] for level in levels], float)
    valid = np.isfinite(error) & (error > np.finfo(float).eps)
    if np.sum(valid) < 2:
        return float("nan")
    return float(np.polyfit(np.log(h[valid]), np.log(error[valid]), 1)[0])


class AnalyticRegistry:
    """Explicit registry; importing Research Studio runs no benchmarks."""

    def __init__(self):
        self._cases: dict[str, tuple[AnalyticCaseSpec, Callable[[], AnalyticCaseResult]]] = {}

    def register(self, spec: AnalyticCaseSpec,
                 runner: Callable[[], AnalyticCaseResult]) -> None:
        if spec.case_id in self._cases:
            raise ValueError(f"duplicate analytic case: {spec.case_id}")
        self._cases[spec.case_id] = (spec, runner)

    @property
    def case_ids(self) -> tuple[str, ...]:
        return tuple(self._cases)

    def spec(self, case_id: str) -> AnalyticCaseSpec:
        return self._cases[case_id][0]

    def run(self, case_id: str) -> AnalyticCaseResult:
        spec, runner = self._cases[case_id]
        result = runner()
        if result.spec != spec:
            raise ValueError(f"runner returned a different specification for {case_id}")
        return result

    def run_all(self) -> dict[str, AnalyticCaseResult]:
        return {case_id: self.run(case_id) for case_id in self.case_ids}


__all__ = ["AnalyticCaseResult", "AnalyticCaseSpec", "AnalyticRegistry",
           "RefinementLevel", "observed_order"]
