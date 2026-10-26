"""Synthetic, reproducible qualification of provider planning behavior."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import time
from typing import Any, Mapping

from ..base import ToolCall, ToolDescriptor
from ..context import InvestigationContext
from ..permissions import AgentPermission, PermissionGrant
from ..planning import BoundedInvestigationPlanner
from .metrics import ProviderCallMetrics


def _signature(call: ToolCall):
    return call.tool, json.dumps(
        dict(call.arguments), sort_keys=True, separators=(",", ":"), allow_nan=False,
    )


@dataclass(frozen=True, slots=True)
class PlanningEvaluationFixture:
    fixture_id: str
    question: str
    case_id: str
    run_ids: tuple[str, ...]
    descriptors: tuple[ToolDescriptor, ...]
    expected_calls: tuple[ToolCall, ...]

    def __post_init__(self):
        if not self.fixture_id or not self.question or not self.case_id:
            raise ValueError("evaluation fixture identity is incomplete")
        object.__setattr__(self, "run_ids", tuple(self.run_ids))
        object.__setattr__(self, "descriptors", tuple(self.descriptors))
        object.__setattr__(self, "expected_calls", tuple(self.expected_calls))
        available = {item.name for item in self.descriptors}
        if not self.expected_calls or any(item.tool not in available for item in self.expected_calls):
            raise ValueError("expected calls must reference available fixture tools")

    def context(self):
        return InvestigationContext(
            f"evaluation_{self.fixture_id}", self.question, self.case_id,
            hashlib.sha256(self.fixture_id.encode()).hexdigest(),
            PermissionGrant(AgentPermission.ANALYZE), self.run_ids,
            attributes={"synthetic_evaluation": True, "fixture_id": self.fixture_id},
        )

    def fingerprint(self):
        value = {
            "fixture_id": self.fixture_id, "question": self.question,
            "case_id": self.case_id, "run_ids": list(self.run_ids),
            "tools": [item.name for item in self.descriptors],
            "expected": [_signature(item) for item in self.expected_calls],
        }
        return hashlib.sha256(json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class PlanningFixtureResult:
    fixture_id: str
    status: str
    latency_seconds: float
    precision: float
    recall: float
    f1: float
    expected_count: int
    planned_count: int
    missing_calls: tuple[tuple[str, str], ...]
    unexpected_calls: tuple[tuple[str, str], ...]
    duplicate_count: int
    plan_hash: str
    metrics: ProviderCallMetrics
    error_type: str = ""
    error: str = ""

    def to_dict(self):
        return {
            "fixture_id": self.fixture_id, "status": self.status,
            "latency_seconds": self.latency_seconds,
            "precision": self.precision, "recall": self.recall, "f1": self.f1,
            "expected_count": self.expected_count, "planned_count": self.planned_count,
            "missing_calls": [list(item) for item in self.missing_calls],
            "unexpected_calls": [list(item) for item in self.unexpected_calls],
            "duplicate_count": self.duplicate_count, "plan_hash": self.plan_hash,
            "metrics": self.metrics.to_dict(),
            "error_type": self.error_type, "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class PlanningEvaluationReport:
    provider: str
    suite_hash: str
    status: str
    results: tuple[PlanningFixtureResult, ...]
    summary: Mapping[str, Any]

    def to_dict(self):
        return {
            "schema_version": 1, "provider": self.provider,
            "suite_hash": self.suite_hash, "status": self.status,
            "results": [item.to_dict() for item in self.results],
            "summary": dict(self.summary),
        }


class PlanningProviderEvaluator:
    def __init__(self, fixtures, *, maximum_calls=32):
        self.fixtures = tuple(fixtures)
        if not self.fixtures:
            raise ValueError("provider evaluation requires fixtures")
        self.maximum_calls = int(maximum_calls)
        self.suite_hash = hashlib.sha256("|".join(
            item.fingerprint() for item in self.fixtures
        ).encode()).hexdigest()

    @staticmethod
    def _metrics(provider):
        take = getattr(provider, "take_metrics", None)
        return take() if callable(take) else ProviderCallMetrics()

    def evaluate(self, provider, *, provider_name):
        results = []
        for fixture in self.fixtures:
            started = time.monotonic()
            try:
                planned = BoundedInvestigationPlanner(
                    provider, provider_name=provider_name,
                    maximum_calls=self.maximum_calls,
                    provider_metadata={"evaluation_fixture": fixture.fixture_id},
                ).plan(fixture.context(), fixture.descriptors)
                actual_values = [_signature(item) for item in planned.plan.calls]
                expected = set(_signature(item) for item in fixture.expected_calls)
                actual = set(actual_values)
                matched = len(expected & actual)
                precision = matched / len(actual) if actual else 0.0
                recall = matched / len(expected)
                f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
                duplicates = len(actual_values) - len(actual)
                passed = precision == 1.0 and recall == 1.0 and duplicates == 0
                results.append(PlanningFixtureResult(
                    fixture.fixture_id, "passed" if passed else "failed",
                    max(0.0, time.monotonic() - started), precision, recall, f1,
                    len(expected), len(actual_values), tuple(sorted(expected - actual)),
                    tuple(sorted(actual - expected)), duplicates, planned.content_hash,
                    self._metrics(provider),
                ))
            except Exception as error:
                results.append(PlanningFixtureResult(
                    fixture.fixture_id, "failed", max(0.0, time.monotonic() - started),
                    0.0, 0.0, 0.0, len(fixture.expected_calls), 0,
                    tuple(sorted(_signature(item) for item in fixture.expected_calls)), (),
                    0, "", self._metrics(provider), type(error).__name__,
                    str(error) or type(error).__name__,
                ))
        passed = sum(item.status == "passed" for item in results)
        token_values = [item.metrics.total_tokens for item in results
                        if item.metrics.total_tokens is not None]
        summary = {
            "fixture_count": len(results), "passed_count": passed,
            "failed_count": len(results) - passed,
            "pass_rate": passed / len(results),
            "mean_f1": sum(item.f1 for item in results) / len(results),
            "total_latency_seconds": sum(item.latency_seconds for item in results),
            "reported_total_tokens": None if not token_values else sum(token_values),
        }
        return PlanningEvaluationReport(
            str(provider_name), self.suite_hash,
            "qualified" if passed == len(results) else "failed",
            tuple(results), summary,
        )


def default_planning_evaluation_fixtures():
    descriptions = {
        "analyze_deck_review": "Inspect deck structure",
        "analyze_grid_quality": "Inspect grid quality",
        "analyze_rock_fluid": "Inspect rock and fluid properties",
        "analyze_initialization": "Inspect initialization",
        "analyze_wells_schedule": "Inspect wells and schedule",
        "analyze_numerical_diagnostics": "Inspect convergence",
        "analyze_reservoir_performance": "Inspect reservoir performance",
        "analyze_conservation": "Inspect conservation",
        "analyze_invariants": "Inspect full-array invariants",
        "compare_reference_runs": "Compare reference and candidate runs",
    }
    descriptor = lambda name: ToolDescriptor(
        name, descriptions[name], AgentPermission.ANALYZE,
    )
    case_tools = tuple(descriptor(name) for name in (
        "analyze_deck_review", "analyze_grid_quality", "analyze_rock_fluid",
        "analyze_initialization", "analyze_wells_schedule",
    ))
    run_tools = tuple(descriptor(name) for name in (
        "analyze_numerical_diagnostics", "analyze_reservoir_performance",
        "analyze_conservation", "analyze_invariants",
    ))
    case_id, run_a, run_b = "synthetic_case", "synthetic_reference", "synthetic_candidate"
    return (
        PlanningEvaluationFixture(
            "case_readiness", "Synthetic fixture: assess whether this case is ready.",
            case_id, (), case_tools,
            tuple(ToolCall(f"expected_case_{i}", item.name, {"case_id": case_id})
                  for i, item in enumerate(case_tools)),
        ),
        PlanningEvaluationFixture(
            "run_diagnostics", "Synthetic fixture: diagnose the selected run.",
            case_id, (run_a,), run_tools,
            tuple(ToolCall(f"expected_run_{i}", item.name, {"run_id": run_a})
                  for i, item in enumerate(run_tools)),
        ),
        PlanningEvaluationFixture(
            "reference_comparison", "Synthetic fixture: compare candidate with reference.",
            case_id, (run_a, run_b), (*run_tools, descriptor("compare_reference_runs")),
            (*tuple(ToolCall(f"expected_a_{i}", item.name, {"run_id": run_a})
                    for i, item in enumerate(run_tools)),
             *tuple(ToolCall(f"expected_b_{i}", item.name, {"run_id": run_b})
                    for i, item in enumerate(run_tools)),
             ToolCall("expected_compare", "compare_reference_runs", {
                 "reference_run_id": run_a, "candidate_run_id": run_b,
             })),
        ),
    )


__all__ = [
    "PlanningEvaluationFixture", "PlanningEvaluationReport",
    "PlanningFixtureResult", "PlanningProviderEvaluator",
    "default_planning_evaluation_fixtures",
]
