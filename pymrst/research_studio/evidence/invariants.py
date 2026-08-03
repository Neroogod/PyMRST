"""Observation/strict/reference invariant profiles with no state mutation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Iterable

import numpy as np


class InvariantProfile(StrEnum):
    OBSERVE = "observe"
    STRICT = "strict"
    REFERENCE = "reference"


@dataclass(frozen=True, slots=True)
class InvariantResult:
    name: str
    passed: bool
    maximum_violation: float
    tolerance: float
    message: str
    locations: tuple[int, ...] = ()

    def to_dict(self):
        return {
            "name": self.name, "passed": self.passed,
            "maximum_violation": self.maximum_violation,
            "tolerance": self.tolerance, "message": self.message,
            "locations": list(self.locations),
        }


class InvariantViolation(RuntimeError):
    def __init__(self, failures: Iterable[InvariantResult]):
        self.failures = tuple(failures)
        super().__init__("; ".join(result.message for result in self.failures))


class InvariantSuite:
    def __init__(self, providers: Iterable[Callable[[Any], InvariantResult]]):
        self.providers = tuple(providers)

    def evaluate(self, state, profile: InvariantProfile | str = "observe"):
        profile = InvariantProfile(profile)
        results = tuple(provider(state) for provider in self.providers)
        failures = tuple(result for result in results if not result.passed)
        if failures and profile is InvariantProfile.STRICT:
            raise InvariantViolation(failures)
        return results


def finite_field(name: str):
    def evaluate(state):
        values = np.asarray(state[name] if isinstance(state, dict) else getattr(state, name))
        bad = np.flatnonzero(~np.isfinite(values))
        return InvariantResult(
            f"finite:{name}", bad.size == 0, float(bad.size), 0.0,
            f"{name} contains {bad.size} non-finite values", tuple(map(int, bad[:20])),
        )
    return evaluate


def bounded_field(name: str, lower: float, upper: float, tolerance: float = 1e-10):
    if lower > upper or tolerance < 0:
        raise ValueError("invalid invariant bounds")

    def evaluate(state):
        values = np.asarray(state[name] if isinstance(state, dict) else getattr(state, name), dtype=float)
        violation = np.maximum(lower-values, values-upper)
        bad = np.flatnonzero(violation > tolerance)
        maximum = float(np.max(violation, initial=0.0))
        return InvariantResult(
            f"bounds:{name}", bad.size == 0, maximum, tolerance,
            f"{name} violates [{lower}, {upper}] by at most {maximum:.6g}",
            tuple(map(int, bad[:20])),
        )
    return evaluate


__all__ = [
    "InvariantProfile", "InvariantResult", "InvariantSuite",
    "InvariantViolation", "bounded_field", "finite_field",
]
