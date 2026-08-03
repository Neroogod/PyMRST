"""M4.2 surrogate schemas, validity domains and fallback policy."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from pymrst.research_studio.experiments.definition import canonical_value


SURROGATE_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TensorField:
    name: str
    unit: str
    shape: tuple[int | None, ...]
    basis: str = "native"
    dtype: str = "float64"

    def __post_init__(self):
        if not self.name or not self.unit or not self.basis:
            raise ValueError("tensor field name, unit and basis are required")
        if any(value is not None and int(value) <= 0 for value in self.shape):
            raise ValueError("tensor dimensions must be positive or dynamic")

    def validate(self, value):
        array = np.asarray(value)
        if array.ndim != len(self.shape):
            raise ValueError(f"{self.name} expects {len(self.shape)} dimensions")
        for actual, expected in zip(array.shape, self.shape):
            if expected is not None and actual != expected:
                raise ValueError(f"{self.name} expects shape {self.shape}, got {array.shape}")
        if array.dtype.kind not in "fiu" or np.any(~np.isfinite(array)):
            raise ValueError(f"{self.name} must contain finite numeric values")
        return array.astype(self.dtype, copy=False)


@dataclass(frozen=True, slots=True)
class ValidityDomain:
    bounds: Mapping[str, tuple[float, float]]
    composition_fields: tuple[str, ...] = ()
    composition_tolerance: float = 1e-8

    def __post_init__(self):
        bounds = {str(name): tuple(map(float, limits))
                  for name, limits in self.bounds.items()}
        if any(len(value) != 2 or not value[0] < value[1] for value in bounds.values()):
            raise ValueError("validity bounds require increasing low/high pairs")
        if self.composition_tolerance < 0:
            raise ValueError("composition tolerance must be nonnegative")
        object.__setattr__(self, "bounds", bounds)
        object.__setattr__(self, "composition_fields",
                           tuple(map(str, self.composition_fields)))

    def check(self, inputs: Mapping[str, np.ndarray]):
        reasons = []
        for name, (low, high) in self.bounds.items():
            if name not in inputs:
                reasons.append(f"missing domain field {name}")
                continue
            value = np.asarray(inputs[name], dtype=float)
            if np.any(value < low) or np.any(value > high):
                reasons.append(f"{name} outside [{low}, {high}]")
        if self.composition_fields:
            missing = [name for name in self.composition_fields if name not in inputs]
            if missing:
                reasons.append(f"missing composition fields {missing}")
            else:
                values = [np.asarray(inputs[name], float) for name in self.composition_fields]
                total = np.sum(values, axis=0)
                if any(np.any(value < -self.composition_tolerance) for value in values):
                    reasons.append("composition contains negative fractions")
                if np.any(np.abs(total-1.0) > self.composition_tolerance):
                    reasons.append("composition does not sum to one")
        return tuple(reasons)


@dataclass(frozen=True, slots=True)
class ReferenceSamplingPolicy:
    fraction: float = 0.0
    minimum_per_batch: int = 0
    seed: int = 0

    def __post_init__(self):
        if not 0 <= self.fraction <= 1 or self.minimum_per_batch < 0 or self.seed < 0:
            raise ValueError("invalid reference sampling policy")


@dataclass(frozen=True, slots=True)
class SurrogateContract:
    surrogate_id: str
    inputs: tuple[TensorField, ...]
    outputs: tuple[TensorField, ...]
    validity: ValidityDomain
    training_data_fingerprint: str
    uncertainty_kind: str
    invariant_names: tuple[str, ...]
    physics_fallback: str
    reference_sampling: ReferenceSamplingPolicy = ReferenceSamplingPolicy()
    devices: tuple[str, ...] = ("cpu",)
    maximum_batch_size: int | None = None
    schema_version: int = SURROGATE_SCHEMA_VERSION

    def __post_init__(self):
        if self.schema_version != SURROGATE_SCHEMA_VERSION:
            raise ValueError("unsupported surrogate schema")
        if not self.surrogate_id or not self.training_data_fingerprint:
            raise ValueError("surrogate id and training fingerprint are required")
        if ":" not in self.physics_fallback:
            raise ValueError("physics fallback must use module:symbol syntax")
        if self.uncertainty_kind not in {"none", "ensemble", "variance", "distance"}:
            raise ValueError("unsupported uncertainty mechanism")
        for fields, label in ((self.inputs, "input"), (self.outputs, "output")):
            names = [value.name for value in fields]
            if not names or len(set(names)) != len(names):
                raise ValueError(f"surrogate {label} names must be nonempty and unique")
        if self.maximum_batch_size is not None and self.maximum_batch_size <= 0:
            raise ValueError("maximum batch size must be positive")
        object.__setattr__(self, "invariant_names", tuple(map(str, self.invariant_names)))
        object.__setattr__(self, "devices", tuple(map(str, self.devices)))

    def validate_inputs(self, values):
        expected = {field.name for field in self.inputs}
        if set(values) != expected:
            raise ValueError(f"surrogate inputs differ: expected {sorted(expected)}")
        normalized = {field.name: field.validate(values[field.name])
                      for field in self.inputs}
        batch = next(iter(normalized.values())).shape[0]
        if any(value.shape[0] != batch for value in normalized.values()):
            raise ValueError("surrogate inputs must share their batch dimension")
        if self.maximum_batch_size is not None and batch > self.maximum_batch_size:
            raise ValueError(f"batch {batch} exceeds maximum {self.maximum_batch_size}")
        return normalized, self.validity.check(normalized)

    def validate_outputs(self, values):
        expected = {field.name for field in self.outputs}
        if set(values) != expected:
            raise ValueError(f"surrogate outputs differ: expected {sorted(expected)}")
        return {field.name: field.validate(values[field.name]) for field in self.outputs}


@dataclass(frozen=True, slots=True)
class SurrogatePrediction:
    outputs: Mapping[str, np.ndarray]
    uncertainty: np.ndarray | None = None
    invariant_failures: tuple[str, ...] = ()


__all__ = ["ReferenceSamplingPolicy", "SURROGATE_SCHEMA_VERSION",
           "SurrogateContract", "SurrogatePrediction", "TensorField",
           "ValidityDomain"]
