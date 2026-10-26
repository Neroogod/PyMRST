"""Acceptance tests for specialist graphs, conflict, uncertainty, and review."""

from __future__ import annotations

import hashlib

import pytest

from pymrst.research_studio.agents import (
    AgentPermission,
    DeterministicOrchestrator,
    GraphInvestigationOutcome,
    HumanReviewDecision,
    InvestigationContext,
    InvestigationGraph,
    InvestigationNode,
    InvestigationStatus,
    PermissionGrant,
    ReviewAction,
    ReviewTarget,
    SingleCaseInvestigator,
    SpecialistGraphInvestigator,
    ToolCall,
    ToolDescriptor,
    ToolResult,
)
from pymrst.research_studio.agents.memory import EntryKind
from pymrst.research_studio.agents.schemas.findings import (
    ClaimKind,
    FindingSeverity,
)
from pymrst.research_studio.agents.specialists import DeterministicSpecialist


DIGEST = hashlib.sha256(b"m4-case").hexdigest()


def context():
    return InvestigationContext(
        "investigation_m4", "Explain pressure behavior", "case_m4", DIGEST,
        PermissionGrant(AgentPermission.ANALYZE, allowed_tools=("pressure_a", "pressure_b")),
        run_ids=("run_m4",),
    )


def evidence(name):
    return {
        "kind": "state", "uri": f"run://run_m4/{name}",
        "label": f"{name} state evidence", "locator": {"cell": 4},
        "checksum": hashlib.sha256(name.encode()).hexdigest(),
    }


class AnalysisTool:
    def __init__(self, name, statement, trace):
        self.descriptor = ToolDescriptor(name, f"Run {name}", AgentPermission.ANALYZE)
        self.statement = statement
        self.trace = trace

    def invoke(self, call):
        self.trace.append(call.tool)
        return ToolResult(call.call_id, True, {
            "analysis_kind": call.tool,
            "evidence": evidence(call.tool),
            "findings": ({
                "category": "reservoir.pressure",
                "severity": "warning",
                "statement": self.statement,
                "affected_objects": ("cell:4",),
            },),
        })


class SpecialistA(DeterministicSpecialist):
    name = "pressure_balance"
    tool_name = "pressure_a"


class SpecialistB(DeterministicSpecialist):
    name = "pressure_performance"
    tool_name = "pressure_b"

    def analyze(self, result):
        item = result.value["findings"][0]
        return (self.make_finding(
            result, category=item["category"], severity=FindingSeverity.WARNING,
            statement=item["statement"], affected_objects=item["affected_objects"],
            claim_kind=ClaimKind.INFERRED, confidence=0.8,
        ),)


def graph():
    # Deliberately declare the dependent node first.
    return InvestigationGraph("pressure_graph", (
        InvestigationNode(
            "performance", ToolCall("call_b", "pressure_b", {}), ("balance",),
        ),
        InvestigationNode("balance", ToolCall("call_a", "pressure_a", {})),
    ))


def coordinator(trace):
    tools = (
        AnalysisTool("pressure_a", "Pressure support is adequate.", trace),
        AnalysisTool("pressure_b", "Pressure support is insufficient.", trace),
    )
    pipeline = SingleCaseInvestigator(
        DeterministicOrchestrator(tools), (SpecialistA(), SpecialistB()),
    )
    return SpecialistGraphInvestigator(pipeline)


def test_graph_validates_dependencies_cycles_and_stable_topological_order():
    assert [node.node_id for level in graph().topological_levels() for node in level] == [
        "balance", "performance",
    ]
    with pytest.raises(ValueError, match="unknown dependencies"):
        InvestigationGraph("bad", (
            InvestigationNode("a", ToolCall("a", "pressure_a", {}), ("missing",)),
        ))
    with pytest.raises(ValueError, match="cycle"):
        InvestigationGraph("cycle", (
            InvestigationNode("a", ToolCall("a", "pressure_a", {}), ("b",)),
            InvestigationNode("b", ToolCall("b", "pressure_b", {}), ("a",)),
        ))


def test_cross_specialist_conflict_waits_for_review_and_preserves_uncertainty():
    trace = []
    outcome = coordinator(trace).investigate(
        context(), graph(), title="Pressure investigation",
    )
    assert trace == ["pressure_a", "pressure_b"]
    assert outcome.investigation.record.status is InvestigationStatus.WAITING_APPROVAL
    assert outcome.investigation.report is not None
    assert not outcome.investigation.report.qualified
    assert len(outcome.conflicts) == 1 and not outcome.conflicts[0].resolved
    assert len(outcome.uncertainties) == 1
    kinds = {entry.kind for entry in outcome.investigation.record.entries}
    assert {EntryKind.EXECUTION_GRAPH, EntryKind.CONFLICT, EntryKind.UNCERTAINTY} <= kinds
    assert "Unresolved conflict" in outcome.investigation.report.limitations[0]
    assert GraphInvestigationOutcome.from_dict(outcome.to_dict()) == outcome


def test_named_human_resolves_conflict_and_approves_qualified_report():
    first = coordinator([]).investigate(context(), graph(), title="Draft")
    conflict = first.conflicts[0]
    reviews = (
        HumanReviewDecision(
            "decision_conflict", ReviewTarget.CONFLICT, conflict.conflict_id,
            ReviewAction.RESOLVE, "reservoir_engineer@example.com",
            "Pressure-balance evidence is the controlling interpretation.",
            selected_finding_ids=(conflict.finding_ids[0],),
        ),
        HumanReviewDecision(
            "decision_report", ReviewTarget.REPORT, context().investigation_id,
            ReviewAction.APPROVE, "asset_manager@example.com",
            "Evidence and stated limitations are acceptable.",
        ),
    )
    outcome = coordinator([]).investigate(
        context(), graph(), title="Qualified pressure investigation",
        reviews=reviews, qualification_requested=True,
    )
    assert outcome.investigation.record.status is InvestigationStatus.COMPLETED
    assert outcome.investigation.report.qualified
    assert outcome.conflicts[0].resolved
    review_section = next(
        section for section in outcome.investigation.report.sections
        if section.section_id == "human_review"
    )
    assert "reservoir_engineer@example.com" in review_section.narrative
    assert "decision_conflict" in review_section.narrative
    assert sum(entry.kind is EntryKind.HUMAN_DECISION
               for entry in outcome.investigation.record.entries) == 2


def test_qualification_never_infers_human_approval():
    outcome = coordinator([]).investigate(
        context(), graph(), title="Needs review", qualification_requested=True,
    )
    # Conflict resolution is needed first, so this remains approval-gated.
    assert outcome.investigation.record.status is InvestigationStatus.WAITING_APPROVAL
    assert not outcome.investigation.report.qualified


def test_review_references_and_actions_fail_closed():
    unknown = HumanReviewDecision(
        "bad", ReviewTarget.FINDING, "missing", ReviewAction.APPROVE,
        "reviewer", "Looks correct.",
    )
    with pytest.raises(ValueError, match="unknown finding"):
        coordinator([]).investigate(
            context(), graph(), title="Bad review", reviews=(unknown,),
        )
    with pytest.raises(ValueError, match="only one human decision"):
        coordinator([]).investigate(
            context(), graph(), title="Duplicate review", reviews=(
                HumanReviewDecision(
                    "one", ReviewTarget.REPORT, context().investigation_id,
                    ReviewAction.APPROVE, "reviewer_a", "Approved.",
                ),
                HumanReviewDecision(
                    "two", ReviewTarget.REPORT, context().investigation_id,
                    ReviewAction.REJECT, "reviewer_b", "Rejected.",
                ),
            ),
        )
