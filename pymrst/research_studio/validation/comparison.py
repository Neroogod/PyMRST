"""Normalized, mapping-aware comparison of simulator histories and fields."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .mappings import EntityMapping, identity_mapping


@dataclass(frozen=True, slots=True)
class ComparisonSeries:
    """A time-by-entity quantity with declared conventions."""

    values: np.ndarray
    times: np.ndarray
    entity_names: tuple[str, ...] = ()
    unit: str = "dimensionless"
    sign_convention: str = "native"

    def __post_init__(self):
        values = np.asarray(self.values, dtype=float)
        times = np.asarray(self.times, dtype=float).reshape(-1)
        if values.ndim == 1:
            values = values[:, None]
        if values.ndim != 2 or values.shape[0] != times.size:
            raise ValueError("values must be time-by-entity and align with times")
        if times.size == 0 or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError("times must be finite, nonempty, and strictly increasing")
        if self.entity_names and len(self.entity_names) != values.shape[1]:
            raise ValueError("entity_names must match the value columns")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "entity_names", tuple(str(v) for v in self.entity_names))


@dataclass(frozen=True, slots=True)
class TolerancePolicy:
    absolute: float = 0.0
    relative: float = 0.0
    scale: float = 1.0

    def __post_init__(self):
        if min(self.absolute, self.relative) < 0.0 or self.scale <= 0.0:
            raise ValueError("tolerances must be nonnegative and scale positive")


@dataclass(frozen=True, slots=True)
class ComparisonResult:
    passed: bool
    maximum_absolute_error: float
    maximum_relative_error: float
    maximum_scaled_error: float
    worst_time: float
    worst_entity: str
    compared_shape: tuple[int, int]
    reference_unit: str
    candidate_unit: str
    unit_scale: float
    sign_scale: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "maximum_absolute_error": self.maximum_absolute_error,
            "maximum_relative_error": self.maximum_relative_error,
            "maximum_scaled_error": self.maximum_scaled_error,
            "worst_time": self.worst_time,
            "worst_entity": self.worst_entity,
            "compared_shape": list(self.compared_shape),
            "reference_unit": self.reference_unit,
            "candidate_unit": self.candidate_unit,
            "unit_scale": self.unit_scale,
            "sign_scale": self.sign_scale,
        }


@dataclass(frozen=True, slots=True)
class NamedAxis:
    name: str
    identities: tuple[str, ...]

    def __post_init__(self):
        if not self.name or not self.identities:
            raise ValueError("named axes require a name and at least one identity")
        identities = tuple(str(value) for value in self.identities)
        if len(set(identities)) != len(identities):
            raise ValueError(f"axis {self.name!r} identities must be unique")
        object.__setattr__(self, "identities", identities)


@dataclass(frozen=True, slots=True)
class NamedFieldSeries:
    """Time history with named, independently mappable field axes."""

    values: np.ndarray
    times: np.ndarray
    axes: tuple[NamedAxis, ...]
    unit: str = "dimensionless"
    sign_convention: str = "native"

    def __post_init__(self):
        values = np.asarray(self.values, dtype=float)
        times = np.asarray(self.times, dtype=float).reshape(-1)
        axes = tuple(self.axes)
        if values.ndim != len(axes)+1 or values.shape[0] != times.size:
            raise ValueError("field shape must be time followed by its declared axes")
        if times.size == 0 or np.any(~np.isfinite(times)) or np.any(np.diff(times) <= 0):
            raise ValueError("times must be finite, nonempty, and strictly increasing")
        names = [axis.name for axis in axes]
        if len(set(names)) != len(names):
            raise ValueError("field axis names must be unique")
        expected = tuple(len(axis.identities) for axis in axes)
        if values.shape[1:] != expected:
            raise ValueError(f"field axes imply shape {expected}, received {values.shape[1:]}")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "axes", axes)


@dataclass(frozen=True, slots=True)
class FieldComparisonResult:
    passed: bool
    maximum_absolute_error: float
    maximum_relative_error: float
    maximum_scaled_error: float
    worst_time: float
    worst_coordinates: tuple[tuple[str, str], ...]
    compared_shape: tuple[int, ...]
    reference_unit: str
    candidate_unit: str
    unit_scale: float
    sign_scale: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "maximum_absolute_error": self.maximum_absolute_error,
            "maximum_relative_error": self.maximum_relative_error,
            "maximum_scaled_error": self.maximum_scaled_error,
            "worst_time": self.worst_time,
            "worst_coordinates": [list(item) for item in self.worst_coordinates],
            "compared_shape": list(self.compared_shape),
            "reference_unit": self.reference_unit,
            "candidate_unit": self.candidate_unit,
            "unit_scale": self.unit_scale,
            "sign_scale": self.sign_scale,
        }


def _aligned_columns(reference: ComparisonSeries, candidate: ComparisonSeries):
    if reference.entity_names:
        if not candidate.entity_names:
            raise ValueError("candidate entity names are required for named reference data")
        candidate_index = {name: index for index, name in enumerate(candidate.entity_names)}
        missing = [name for name in reference.entity_names if name not in candidate_index]
        if missing:
            raise ValueError(f"candidate is missing entities: {missing}")
        indices = [candidate_index[name] for name in reference.entity_names]
        names = reference.entity_names
    else:
        if reference.values.shape[1] != candidate.values.shape[1]:
            raise ValueError("unnamed series must have the same entity count")
        indices = list(range(reference.values.shape[1]))
        names = tuple(str(index) for index in indices)
    return candidate.values[:, indices], names


def _interpolate(times, values, target_times):
    if target_times[0] < times[0] or target_times[-1] > times[-1]:
        raise ValueError("candidate time range does not cover the reference range")
    return np.column_stack([
        np.interp(target_times, times, values[:, column])
        for column in range(values.shape[1])
    ])


def compare_series(
    reference: ComparisonSeries,
    candidate: ComparisonSeries,
    *,
    tolerance: TolerancePolicy = TolerancePolicy(),
    unit_scale: float = 1.0,
    sign_scale: float | None = None,
    entity_mapping: EntityMapping | None = None,
) -> ComparisonResult:
    """Align candidate to reference, normalize conventions, and compare.

    ``unit_scale`` converts candidate values into reference units. If
    ``sign_scale`` is omitted, equal sign conventions imply ``+1`` and
    different conventions are rejected rather than guessed.
    """
    if not np.isfinite(unit_scale) or unit_scale == 0.0:
        raise ValueError("unit_scale must be finite and nonzero")
    if sign_scale is None:
        if reference.sign_convention != candidate.sign_convention:
            raise ValueError("sign conventions differ; provide an explicit sign_scale")
        sign_scale = 1.0
    if sign_scale not in (-1.0, 1.0):
        raise ValueError("sign_scale must be +1 or -1")
    if entity_mapping is None:
        columns, names = _aligned_columns(reference, candidate)
        reference_values = reference.values
    else:
        if entity_mapping.reference_size != reference.values.shape[1]:
            raise ValueError("mapping reference size does not match comparison series")
        if entity_mapping.candidate_size != candidate.values.shape[1]:
            raise ValueError("mapping candidate size does not match comparison series")
        columns = entity_mapping.align_candidate(candidate.values, axis=1)
        reference_values = entity_mapping.select_reference(reference.values, axis=1)
        names = entity_mapping.identities
    aligned = _interpolate(candidate.times, columns, reference.times)
    normalized = aligned*unit_scale*sign_scale
    if np.any(~np.isfinite(reference_values)) or np.any(~np.isfinite(normalized)):
        raise ValueError("comparison values must be finite")
    absolute = np.abs(normalized-reference_values)
    denominator = np.maximum(np.abs(reference_values), tolerance.scale)
    relative = absolute/denominator
    threshold = tolerance.absolute + tolerance.relative*np.abs(reference_values)
    passed = bool(np.all(absolute <= threshold))
    scaled = absolute/tolerance.scale
    worst_flat = int(np.argmax(scaled))
    row, column = np.unravel_index(worst_flat, scaled.shape)
    return ComparisonResult(
        passed=passed,
        maximum_absolute_error=float(np.max(absolute)),
        maximum_relative_error=float(np.max(relative)),
        maximum_scaled_error=float(np.max(scaled)),
        worst_time=float(reference.times[row]),
        worst_entity=names[column],
        compared_shape=absolute.shape,
        reference_unit=reference.unit,
        candidate_unit=candidate.unit,
        unit_scale=float(unit_scale),
        sign_scale=float(sign_scale),
    )


def compare_named_fields(
    reference: NamedFieldSeries,
    candidate: NamedFieldSeries,
    *,
    tolerance: TolerancePolicy = TolerancePolicy(),
    mappings: dict[str, EntityMapping] | None = None,
    unit_scale: float = 1.0,
    sign_scale: float | None = None,
) -> FieldComparisonResult:
    """Compare arbitrary named field axes after explicit axis alignment."""
    if not np.isfinite(unit_scale) or unit_scale == 0.0:
        raise ValueError("unit_scale must be finite and nonzero")
    if sign_scale is None:
        if reference.sign_convention != candidate.sign_convention:
            raise ValueError("sign conventions differ; provide an explicit sign_scale")
        sign_scale = 1.0
    if sign_scale not in (-1.0, 1.0):
        raise ValueError("sign_scale must be +1 or -1")
    reference_names = tuple(axis.name for axis in reference.axes)
    candidate_lookup = {axis.name: index for index, axis in enumerate(candidate.axes)}
    if set(candidate_lookup) != set(reference_names):
        raise ValueError("reference and candidate field axis names differ")
    permutation = (0,)+tuple(candidate_lookup[name]+1 for name in reference_names)
    candidate_values = np.transpose(candidate.values, permutation)
    reference_values = reference.values
    declared = {} if mappings is None else dict(mappings)
    unknown = sorted(set(declared)-set(reference_names))
    if unknown:
        raise ValueError(f"mappings supplied for unknown axes: {unknown}")
    aligned_identities: list[tuple[str, ...]] = []
    for position, (name, reference_axis) in enumerate(zip(reference_names, reference.axes), start=1):
        candidate_axis = candidate.axes[candidate_lookup[name]]
        mapping = declared.get(name)
        if mapping is None:
            mapping = identity_mapping(
                reference_axis.identities, candidate_axis.identities,
                kind=f"field_axis:{name}",
            )
        if (mapping.reference_size != len(reference_axis.identities)
                or mapping.candidate_size != len(candidate_axis.identities)):
            raise ValueError(f"mapping sizes do not match field axis {name!r}")
        reference_values = mapping.select_reference(reference_values, axis=position)
        candidate_values = mapping.align_candidate(candidate_values, axis=position)
        aligned_identities.append(mapping.identities)
    flat_candidate = candidate_values.reshape(candidate_values.shape[0], -1)
    flat_aligned = _interpolate(candidate.times, flat_candidate, reference.times)
    aligned = flat_aligned.reshape((reference.times.size,)+candidate_values.shape[1:])
    normalized = aligned*unit_scale*sign_scale
    if np.any(~np.isfinite(reference_values)) or np.any(~np.isfinite(normalized)):
        raise ValueError("comparison values must be finite")
    absolute = np.abs(normalized-reference_values)
    relative = absolute/np.maximum(np.abs(reference_values), tolerance.scale)
    scaled = absolute/tolerance.scale
    threshold = tolerance.absolute+tolerance.relative*np.abs(reference_values)
    flat_index = int(np.argmax(scaled))
    location = np.unravel_index(flat_index, scaled.shape)
    coordinates = tuple(
        (name, aligned_identities[index][location[index+1]])
        for index, name in enumerate(reference_names)
    )
    return FieldComparisonResult(
        passed=bool(np.all(absolute <= threshold)),
        maximum_absolute_error=float(np.max(absolute)),
        maximum_relative_error=float(np.max(relative)),
        maximum_scaled_error=float(np.max(scaled)),
        worst_time=float(reference.times[location[0]]),
        worst_coordinates=coordinates,
        compared_shape=absolute.shape,
        reference_unit=reference.unit, candidate_unit=candidate.unit,
        unit_scale=float(unit_scale), sign_scale=float(sign_scale),
    )


__all__ = [
    "ComparisonResult", "ComparisonSeries", "FieldComparisonResult",
    "NamedAxis", "NamedFieldSeries", "TolerancePolicy", "compare_named_fields",
    "compare_series",
]
