"""Qualification tests for synthetic provider-planning evaluations."""

from pymrst.research_studio.agents.orchestrator import InvestigationPlan
from pymrst.research_studio.agents.planning import DeterministicPlanningProvider
from pymrst.research_studio.agents.providers import (
    PlanningProviderEvaluator, ProviderCallMetrics,
    default_planning_evaluation_fixtures,
)


def test_deterministic_planner_qualifies_on_reproducible_synthetic_suite():
    evaluator = PlanningProviderEvaluator(default_planning_evaluation_fixtures())
    first = evaluator.evaluate(DeterministicPlanningProvider(), provider_name="deterministic-v1")
    second = evaluator.evaluate(DeterministicPlanningProvider(), provider_name="deterministic-v1")
    assert first.status == "qualified"
    assert first.summary["pass_rate"] == 1.0
    assert first.summary["mean_f1"] == 1.0
    assert first.summary["reported_total_tokens"] is None
    assert first.suite_hash == second.suite_hash
    assert [item.plan_hash for item in first.results] == [
        item.plan_hash for item in second.results
    ]


class EmptyProvider:
    def complete(self, messages, tools, response_type):
        # Returning the wrong type is a provider contract failure, not a crash.
        return {"calls": []}


def test_contract_failures_are_retained_as_failed_fixture_results():
    report = PlanningProviderEvaluator(
        default_planning_evaluation_fixtures()[:1],
    ).evaluate(EmptyProvider(), provider_name="broken")
    assert report.status == "failed"
    assert report.results[0].error_type == "TypeError"
    assert report.summary["failed_count"] == 1


class CountingDeterministicProvider(DeterministicPlanningProvider):
    def take_metrics(self):
        return ProviderCallMetrics(100, 25, 125)


def test_report_aggregates_provider_reported_tokens_without_prompt_retention():
    report = PlanningProviderEvaluator(
        default_planning_evaluation_fixtures()[:2],
    ).evaluate(CountingDeterministicProvider(), provider_name="counting")
    assert report.status == "qualified"
    assert report.summary["reported_total_tokens"] == 250
    assert all(item.metrics.total_tokens == 125 for item in report.results)
    assert "question" not in report.to_dict()["results"][0]

