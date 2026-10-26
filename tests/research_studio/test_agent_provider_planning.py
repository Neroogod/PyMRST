"""Qualification for provider-independent, bounded investigation planning."""

import hashlib

import pytest

from pymrst.research_studio.agents import (
    AgentPermission, BoundedInvestigationPlanner, DeterministicPlanningProvider,
    InvestigationContext, InvestigationPlan, PermissionGrant, ToolCall,
    ToolDescriptor,
)


def context():
    return InvestigationContext(
        "investigation_plan", "Compare convergence", "case_plan",
        hashlib.sha256(b"case").hexdigest(),
        PermissionGrant(AgentPermission.ANALYZE), ("run_a", "run_b"),
    )


def descriptors():
    return tuple(ToolDescriptor(name, name.replace("_", " "), AgentPermission.ANALYZE)
                 for name in (
                     "analyze_deck_review", "analyze_grid_quality", "analyze_rock_fluid",
                     "analyze_initialization", "analyze_wells_schedule",
                     "analyze_numerical_diagnostics", "analyze_reservoir_performance",
                     "analyze_conservation", "analyze_invariants", "compare_reference_runs",
                 ))


def test_deterministic_provider_builds_reproducible_context_bound_plan():
    planner = BoundedInvestigationPlanner(
        DeterministicPlanningProvider(), provider_name="deterministic-v1",
    )
    first = planner.plan(context(), descriptors())
    second = planner.plan(context(), descriptors())
    assert first == second
    assert len(first.plan.calls) == 14
    comparison = first.plan.calls[-1]
    assert comparison.arguments == {
        "reference_run_id": "run_a", "candidate_run_id": "run_b",
    }
    assert len(first.content_hash) == 64


class Provider:
    def __init__(self, plan): self.plan_value = plan
    def complete(self, messages, tools, response_type): return self.plan_value


def test_provider_cannot_escape_case_run_tool_or_execution_boundaries():
    cases = (
        (ToolCall("x", "missing", {}), ValueError, "unavailable"),
        (ToolCall("x", "analyze_conservation", {"run_id": "run_other"}), ValueError, "out-of-context"),
        (ToolCall("x", "analyze_grid_quality", {"case_id": "other"}), ValueError, "out-of-context"),
    )
    for call, exception, match in cases:
        planner = BoundedInvestigationPlanner(Provider(InvestigationPlan((call,))))
        with pytest.raises(exception, match=match):
            planner.plan(context(), descriptors())
    workload = ToolDescriptor(
        "run_case", "start run", AgentPermission.RUN_SIMULATION,
        starts_workload=True,
    )
    grant = InvestigationContext(
        "investigation_workload", "try workload", "case_plan",
        hashlib.sha256(b"case").hexdigest(),
        PermissionGrant(AgentPermission.RUN_SIMULATION),
    )
    planner = BoundedInvestigationPlanner(Provider(InvestigationPlan((
        ToolCall("run", "run_case", {}),
    ))))
    with pytest.raises(PermissionError, match="mutations or workloads"):
        planner.plan(grant, (workload,))
