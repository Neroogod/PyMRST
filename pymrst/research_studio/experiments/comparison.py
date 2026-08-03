"""Scenario comparison over normalized member-result categories."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

import numpy as np

from pymrst.research_studio.validation import (
    ComparisonResult, ComparisonSeries, EntityMapping, TolerancePolicy,
    compare_series,
)
from .execution import LazyMemberResult


SCENARIO_COMPARISON_SCHEMA_VERSION = 1
RESULT_CATEGORIES = ("fields", "regions", "wells", "convergence", "conservation")


@dataclass(frozen=True, slots=True)
class ScenarioQuantityPolicy:
    tolerance: TolerancePolicy
    unit: str = "dimensionless"
    sign_convention: str = "native"
    unit_scale: float = 1.0
    sign_scale: float | None = None


@dataclass(frozen=True, slots=True)
class ScenarioComparison:
    reference_member: str
    candidate_member: str
    comparison_group: str
    results: Mapping[str, ComparisonResult]
    schema_version: int = SCENARIO_COMPARISON_SCHEMA_VERSION

    @property
    def passed(self):
        return all(result.passed for result in self.results.values())

    def to_dict(self):
        return {"schema_version": self.schema_version,
                "reference_member": self.reference_member,
                "candidate_member": self.candidate_member,
                "comparison_group": self.comparison_group,
                "passed": self.passed,
                "results": {name: result.to_dict() for name, result in self.results.items()}}


def _series(result: LazyMemberResult, category: str, name: str,
            policy: ScenarioQuantityPolicy):
    values = result.array(category, name)
    flattened = values.reshape(values.shape[0], -1)
    names = tuple(f"{category}.{name}[{index}]" for index in range(flattened.shape[1]))
    return ComparisonSeries(flattened, result.times, names, unit=policy.unit,
                            sign_convention=policy.sign_convention)


def compare_scenarios(reference: LazyMemberResult, candidate: LazyMemberResult, *,
                      policies: Mapping[str, ScenarioQuantityPolicy],
                      mappings: Mapping[str, EntityMapping] | None = None,
                      comparison_group: str = "default") -> ScenarioComparison:
    """Compare category-qualified names such as ``fields.pressure``."""
    mappings = {} if mappings is None else dict(mappings)
    results = {}
    for qualified, policy in policies.items():
        try:
            category, name = qualified.split(".", 1)
        except ValueError as error:
            raise ValueError("scenario policy names must be category.quantity") from error
        if category not in RESULT_CATEGORIES:
            raise ValueError(f"unsupported scenario category: {category}")
        if name not in reference.names(category) or name not in candidate.names(category):
            raise KeyError(f"quantity {qualified!r} is unavailable in both scenarios")
        results[qualified] = compare_series(
            _series(reference, category, name, policy),
            _series(candidate, category, name, policy),
            tolerance=policy.tolerance, unit_scale=policy.unit_scale,
            sign_scale=policy.sign_scale, entity_mapping=mappings.get(qualified),
        )
    reference_id = reference.record["coordinate"]["member_id"]
    candidate_id = candidate.record["coordinate"]["member_id"]
    return ScenarioComparison(reference_id, candidate_id, comparison_group, results)


def compare_performance(reference: LazyMemberResult, candidate: LazyMemberResult, *,
                        tolerances: Mapping[str, TolerancePolicy],
                        comparison_group="performance") -> ScenarioComparison:
    results = {}
    for name, tolerance in tolerances.items():
        if name not in reference.record["performance"] or name not in candidate.record["performance"]:
            raise KeyError(f"performance metric {name!r} is missing")
        ref = ComparisonSeries([[reference.record["performance"][name]]], [1.0], (name,))
        cand = ComparisonSeries([[candidate.record["performance"][name]]], [1.0], (name,))
        results[f"performance.{name}"] = compare_series(ref, cand, tolerance=tolerance)
    return ScenarioComparison(
        reference.record["coordinate"]["member_id"],
        candidate.record["coordinate"]["member_id"], comparison_group, results)


__all__ = ["RESULT_CATEGORIES", "SCENARIO_COMPARISON_SCHEMA_VERSION",
           "ScenarioComparison", "ScenarioQuantityPolicy", "compare_performance",
           "compare_scenarios"]
