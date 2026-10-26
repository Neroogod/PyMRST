"""Interpret checksum-bound full-array run comparisons for agent investigations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ..base import ToolCall, ToolDescriptor, ToolResult
from ..permissions import AgentPermission


@dataclass(frozen=True, slots=True)
class ApplicationComparisonSnapshot:
    comparison_id: str
    reference_run_id: str
    candidate_run_id: str
    checksum: str
    summary: Mapping[str, Any]

    def __post_init__(self):
        if not self.comparison_id or len(self.checksum) != 64:
            raise ValueError("application comparison identity is incomplete")
        object.__setattr__(self, "summary", dict(self.summary))


class ApplicationComparisonCatalog:
    def __init__(self): self._items = {}
    def register(self, item):
        existing = self._items.get(item.comparison_id)
        if existing is not None and existing != item:
            raise ValueError("comparison evidence changed within one investigation")
        self._items[item.comparison_id] = item
        return item
    def get(self, comparison_id):
        try: return self._items[str(comparison_id)]
        except KeyError as error: raise KeyError(f"unknown application comparison: {comparison_id}") from error


class AnalyzeApplicationComparisonTool:
    descriptor = ToolDescriptor(
        "compare_reference_runs", "Interpret a bounded full-array application run comparison",
        AgentPermission.ANALYZE,
    )
    def __init__(self, catalog): self.catalog = catalog
    def invoke(self, call):
        item = self.catalog.get(call.arguments["comparison_id"])
        tolerances = dict(call.arguments.get("tolerances", {}))
        findings, comparisons = [], {}
        for name, metrics in sorted(item.summary.get("fields", {}).items()):
            absolute = float(metrics["maximum_absolute_error"])
            relative = float(metrics["maximum_relative_error"])
            policy = dict(tolerances.get(name, {}))
            has_policy = "absolute" in policy or "relative" in policy
            absolute_limit = float(policy.get("absolute", float("inf")))
            relative_limit = float(policy.get("relative", float("inf")))
            passed = absolute <= absolute_limit and relative <= relative_limit
            comparisons[name] = dict(metrics, passed=passed if has_policy else None)
            severity = "error" if has_policy and not passed else "info"
            qualifier = (
                "exceeds the configured comparison tolerance"
                if has_policy and not passed else
                "is within the configured comparison tolerance"
                if has_policy else "was recorded without an acceptance tolerance"
            )
            findings.append({
                "severity": severity, "category": f"simulator_disagreement.{name}",
                "statement": f"{name}: maximum absolute difference {absolute:.6g}, "
                             f"maximum relative difference {relative:.6g}; {qualifier}.",
                "affected_objects": [f"field:{name}",
                                     f"run:{item.reference_run_id}",
                                     f"run:{item.candidate_run_id}"],
            })
        return ToolResult(call.call_id, True, {
            "analysis_kind": self.descriptor.name,
            "reference_run_id": item.reference_run_id,
            "candidate_run_id": item.candidate_run_id,
            "comparisons": comparisons, "findings": findings,
            "evidence": {
                "kind": "validation", "uri": f"comparison://{item.comparison_id}",
                "label": f"Full-array comparison {item.reference_run_id} → {item.candidate_run_id}",
                "locator": {"common_report_steps": item.summary.get("common_report_steps", [])},
                "checksum": item.checksum,
            },
        })


__all__ = [
    "AnalyzeApplicationComparisonTool", "ApplicationComparisonCatalog",
    "ApplicationComparisonSnapshot",
]
