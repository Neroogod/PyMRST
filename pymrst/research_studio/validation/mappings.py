"""Explicit, serializable mappings between simulator result conventions.

Mappings always answer one question: which candidate entries correspond to
the reference entries, in reference order?  Missing or duplicate identities
are errors.  No cell, phase, component, or well ordering is guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


MAPPING_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RegionAggregation:
    values: np.ndarray
    region_ids: tuple[str, ...]
    reduction: str

    def __post_init__(self):
        array = np.asarray(self.values, dtype=float)
        if array.shape[-1] != len(self.region_ids):
            raise ValueError("region IDs must align with the last value axis")
        object.__setattr__(self, "values", array)
        object.__setattr__(self, "region_ids", tuple(str(v) for v in self.region_ids))


def _unique(values: Sequence[Any], owner: str) -> tuple[Any, ...]:
    result = tuple(values)
    duplicates = sorted({value for value in result if result.count(value) > 1}, key=str)
    if duplicates:
        raise ValueError(f"{owner} identities are not unique: {duplicates}")
    return result


@dataclass(frozen=True, slots=True)
class EntityMapping:
    """Candidate-to-reference index mapping with provenance."""

    kind: str
    reference_indices: np.ndarray
    candidate_indices: np.ndarray
    identities: tuple[str, ...]
    reference_size: int
    candidate_size: int
    schema_version: int = MAPPING_SCHEMA_VERSION

    def __post_init__(self):
        reference = np.asarray(self.reference_indices, dtype=np.int64).reshape(-1)
        candidate = np.asarray(self.candidate_indices, dtype=np.int64).reshape(-1)
        if self.schema_version != MAPPING_SCHEMA_VERSION:
            raise ValueError("unsupported EntityMapping schema version")
        if reference.size != candidate.size or reference.size != len(self.identities):
            raise ValueError("mapping indices and identities must have equal length")
        if self.reference_size < 0 or self.candidate_size < 0:
            raise ValueError("mapping sizes must be nonnegative")
        if (np.any(reference < 0) or np.any(reference >= self.reference_size)
                or np.any(candidate < 0) or np.any(candidate >= self.candidate_size)):
            raise ValueError("mapping index is outside its declared source size")
        if np.unique(reference).size != reference.size or np.unique(candidate).size != candidate.size:
            raise ValueError("mapping indices must be one-to-one")
        object.__setattr__(self, "reference_indices", reference)
        object.__setattr__(self, "candidate_indices", candidate)
        object.__setattr__(self, "identities", tuple(str(value) for value in self.identities))

    @property
    def complete(self) -> bool:
        return self.reference_indices.size == self.reference_size

    def align_candidate(self, values: Any, *, axis: int = -1) -> np.ndarray:
        array = np.asarray(values)
        axis = axis + array.ndim if axis < 0 else axis
        if axis < 0 or axis >= array.ndim:
            raise ValueError("mapping axis is outside the candidate array")
        if array.shape[axis] != self.candidate_size:
            raise ValueError(
                f"candidate axis has size {array.shape[axis]}, expected {self.candidate_size}"
            )
        return np.take(array, self.candidate_indices, axis=axis)

    def select_reference(self, values: Any, *, axis: int = -1) -> np.ndarray:
        array = np.asarray(values)
        axis = axis + array.ndim if axis < 0 else axis
        if axis < 0 or axis >= array.ndim:
            raise ValueError("mapping axis is outside the reference array")
        if array.shape[axis] != self.reference_size:
            raise ValueError(
                f"reference axis has size {array.shape[axis]}, expected {self.reference_size}"
            )
        return np.take(array, self.reference_indices, axis=axis)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "reference_indices": self.reference_indices.tolist(),
            "candidate_indices": self.candidate_indices.tolist(),
            "identities": list(self.identities),
            "reference_size": self.reference_size,
            "candidate_size": self.candidate_size,
            "complete": self.complete,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EntityMapping":
        allowed = {
            "schema_version", "kind", "reference_indices", "candidate_indices",
            "identities", "reference_size", "candidate_size", "complete",
        }
        if set(value) != allowed:
            raise ValueError("invalid EntityMapping fields")
        result = cls(
            kind=str(value["kind"]),
            reference_indices=np.asarray(value["reference_indices"], dtype=np.int64),
            candidate_indices=np.asarray(value["candidate_indices"], dtype=np.int64),
            identities=tuple(value["identities"]),
            reference_size=int(value["reference_size"]),
            candidate_size=int(value["candidate_size"]),
            schema_version=int(value["schema_version"]),
        )
        if bool(value["complete"]) != result.complete:
            raise ValueError("serialized mapping completeness is inconsistent")
        return result


def identity_mapping(
    reference: Sequence[Any],
    candidate: Sequence[Any],
    *,
    kind: str,
    aliases: Mapping[Any, Any] | None = None,
    require_complete: bool = True,
) -> EntityMapping:
    """Map explicit identities without assuming equal ordering.

    ``aliases`` maps candidate identities to reference identities and is
    applied before uniqueness checks. It makes phase/component convention
    changes reviewable instead of embedding a global alias table.
    """
    ref = _unique(reference, "reference")
    alias = {} if aliases is None else dict(aliases)
    cand_raw = tuple(candidate)
    cand = _unique(tuple(alias.get(value, value) for value in cand_raw), "candidate")
    lookup = {value: index for index, value in enumerate(cand)}
    missing = [value for value in ref if value not in lookup]
    if require_complete and missing:
        raise ValueError(f"candidate is missing {kind} identities: {missing}")
    matched = [(index, value) for index, value in enumerate(ref) if value in lookup]
    return EntityMapping(
        kind=kind,
        reference_indices=np.array([index for index, _ in matched], dtype=np.int64),
        candidate_indices=np.array([lookup[value] for _, value in matched], dtype=np.int64),
        identities=tuple(str(value) for _, value in matched),
        reference_size=len(ref),
        candidate_size=len(cand_raw),
    )


def named_entity_mapping(reference_names, candidate_names, **kwargs) -> EntityMapping:
    return identity_mapping(reference_names, candidate_names, kind="named_entity", **kwargs)


def active_global_cell_mapping(
    reference_global_cells: Iterable[int],
    candidate_global_cells: Iterable[int],
    *,
    require_complete: bool = True,
) -> EntityMapping:
    reference = tuple(int(value) for value in reference_global_cells)
    candidate = tuple(int(value) for value in candidate_global_cells)
    return identity_mapping(
        reference, candidate, kind="active_global_cell",
        require_complete=require_complete,
    )


def phase_mapping(reference_phases, candidate_phases, **kwargs) -> EntityMapping:
    return identity_mapping(reference_phases, candidate_phases, kind="phase", **kwargs)


def component_mapping(reference_components, candidate_components, **kwargs) -> EntityMapping:
    return identity_mapping(
        reference_components, candidate_components, kind="component", **kwargs,
    )


def aggregate_regions(
    values: Any,
    regions: Sequence[Any],
    *,
    axis: int = -1,
    reduction: str = "sum",
    weights: Any = None,
    region_order: Sequence[Any] | None = None,
) -> RegionAggregation:
    """Aggregate a cell axis into explicitly identified regions."""
    array = np.asarray(values, dtype=float)
    axis = axis + array.ndim if axis < 0 else axis
    if axis < 0 or axis >= array.ndim:
        raise ValueError("region axis is outside the value array")
    region_values = np.asarray(regions).reshape(-1)
    if array.shape[axis] != region_values.size:
        raise ValueError("regions must contain one identity per cell")
    unique = tuple(sorted(set(region_values.tolist()), key=str))
    order = unique if region_order is None else _unique(tuple(region_order), "region_order")
    unknown = [region for region in order if region not in unique]
    missing = [region for region in unique if region not in order]
    if unknown or missing:
        raise ValueError(f"region_order mismatch: unknown={unknown}, missing={missing}")
    if reduction not in {"sum", "mean", "weighted_mean"}:
        raise ValueError(f"unsupported region reduction: {reduction}")
    weight_array = None
    if reduction == "weighted_mean":
        if weights is None:
            raise ValueError("weighted_mean requires cell weights")
        weight_array = np.asarray(weights, dtype=float).reshape(-1)
        if (weight_array.size != region_values.size or np.any(~np.isfinite(weight_array))
                or np.any(weight_array < 0.0)):
            raise ValueError("region weights must be finite, nonnegative, and cell-aligned")
    moved = np.moveaxis(array, axis, -1)
    outputs = []
    for region in order:
        mask = region_values == region
        selected = moved[..., mask]
        if reduction == "sum":
            output = selected.sum(axis=-1)
        elif reduction == "mean":
            output = selected.mean(axis=-1)
        else:
            selected_weights = weight_array[mask]
            denominator = float(selected_weights.sum())
            if denominator <= 0.0:
                raise ValueError(f"region {region!r} has zero total weight")
            output = np.sum(selected*selected_weights, axis=-1)/denominator
        outputs.append(output)
    return RegionAggregation(
        np.stack(outputs, axis=-1), tuple(str(region) for region in order), reduction,
    )


__all__ = [
    "MAPPING_SCHEMA_VERSION", "EntityMapping", "RegionAggregation",
    "active_global_cell_mapping", "aggregate_regions", "component_mapping",
    "identity_mapping", "named_entity_mapping", "phase_mapping",
]
