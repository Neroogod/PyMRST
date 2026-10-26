"""Acceptance tests for the evidence-qualified single-case agent pipeline."""

from __future__ import annotations

import hashlib

import pytest

from pymrst.research_studio.agents import (
    AgentPermission,
    DeterministicOrchestrator,
    InvestigationContext,
    InvestigationOutcome,
    InvestigationPlan,
    InvestigationStatus,
    PermissionGrant,
    SingleCaseInvestigator,
    ToolCall,
    ToolDescriptor,
    ToolResult,
    read_investigation_bundle,
    write_investigation_bundle,
)
from pymrst.research_studio.agents.memory import EntryKind
from pymrst.research_studio.agents.policies.evidence import EvidencePolicy
from pymrst.research_studio.agents.schemas.findings import FindingSeverity
from pymrst.research_studio.agents.specialists import DeterministicSpecialist
from pymrst.research_studio.agents.tools.report_tools import (
    EngineeringReportBuilder,
    PublishEngineeringReportTool,
    ReportCatalog,
    RenderEngineeringReportTool,
    render_markdown,
)


DIGEST = hashlib.sha256(b"agent-m3-case").hexdigest()
EVIDENCE = {
    "kind": "convergence",
    "uri": "run://run_m3/convergence",
    "label": "Stored convergence history",
    "locator": {"report_step": 4},
    "checksum": hashlib.sha256(b"convergence").hexdigest(),
}


def context(permission=AgentPermission.ANALYZE, allowed=("diagnose",)):
    return InvestigationContext(
        "investigation_m3", "Why did convergence slow?", "case_m3", DIGEST,
        PermissionGrant(permission, allowed_tools=allowed), run_ids=("run_m3",),
    )


class DiagnosticTool:
    descriptor = ToolDescriptor(
        "diagnose", "Inspect stored convergence evidence", AgentPermission.ANALYZE,
    )

    def __init__(self):
        self.invocations = 0

    def invoke(self, call):
        self.invocations += 1
        return ToolResult(call.call_id, True, {
            "analysis_kind": "diagnose",
            "evidence": EVIDENCE,
            "findings": ({
                "category": "numerical.convergence",
                "severity": "warning",
                "statement": "Newton iterations increased at report step 4.",
                "affected_objects": ("report_step:4",),
            },),
        })


class DiagnosticSpecialist(DeterministicSpecialist):
    name = "numerical_diagnostics"
    tool_name = "diagnose"


class WeakSpecialist(DiagnosticSpecialist):
    def analyze(self, result):
        return (self.make_finding(
            result, category="numerical.inference",
            severity=FindingSeverity.WARNING,
            statement="The slowdown may be related to a control transition.",
            confidence=0.2,
        ),)


class DuplicateSpecialist(DiagnosticSpecialist):
    def analyze(self, result):
        finding = super().analyze(result)[0]
        return finding, finding


def plan():
    return InvestigationPlan((ToolCall("call_diagnose", "diagnose", {}),))


def test_single_case_pipeline_runs_once_and_persists_claim_lineage():
    tool = DiagnosticTool()
    pipeline = SingleCaseInvestigator(
        DeterministicOrchestrator((tool,)), (DiagnosticSpecialist(),),
    )
    outcome = pipeline.investigate(
        context(), plan(), title="M3 convergence review",
        qualification_passed=True,
        assumptions=("Stored convergence data are complete.",),
    )

    assert tool.invocations == 1
    assert outcome.record.status is InvestigationStatus.COMPLETED
    assert outcome.report is not None and outcome.report.qualified
    assert len(outcome.findings) == 1
    kinds = tuple(entry.kind for entry in outcome.record.entries)
    assert kinds.count(EntryKind.TOOL_CALL) == 1
    assert kinds.count(EntryKind.TOOL_RESULT) == 1
    assert kinds.count(EntryKind.FINDING) == 1
    assert kinds.count(EntryKind.POLICY_DECISION) == 2
    assert kinds[-1] is EntryKind.REPORT
    restored = InvestigationOutcome.from_dict(outcome.to_dict())
    assert restored == outcome
    assert restored.report.content_hash == outcome.report.content_hash


def test_weak_evidence_is_recorded_but_blocks_report_generation():
    pipeline = SingleCaseInvestigator(
        DeterministicOrchestrator((DiagnosticTool(),)), (WeakSpecialist(),),
        evidence_policy=EvidencePolicy(minimum_calculated_confidence=0.95),
    )
    outcome = pipeline.investigate(context(), plan(), title="Weak review")
    assert outcome.record.status is InvestigationStatus.UNQUALIFIED
    assert outcome.report is None
    assert len(outcome.findings) == 1
    assert "below" in outcome.failure_reason
    assert any(entry.kind is EntryKind.POLICY_DECISION
               for entry in outcome.record.entries)


def test_missing_specialist_and_duplicate_finding_ids_fail_closed():
    missing = SingleCaseInvestigator(
        DeterministicOrchestrator((DiagnosticTool(),)), (),
    ).investigate(context(), plan(), title="Missing specialist")
    assert missing.record.status is InvestigationStatus.FAILED
    assert "no specialist" in missing.failure_reason

    duplicate = SingleCaseInvestigator(
        DeterministicOrchestrator((DiagnosticTool(),)), (DuplicateSpecialist(),),
    ).investigate(context(), plan(), title="Duplicate finding")
    assert duplicate.record.status is InvestigationStatus.FAILED
    assert "duplicate finding" in duplicate.failure_reason


def test_report_builder_and_renderer_are_deterministic():
    result = DiagnosticTool().invoke(ToolCall("call", "diagnose", {}))
    findings = DiagnosticSpecialist().analyze(result)
    builder = EngineeringReportBuilder()
    first = builder.build(context(), findings, title="Deterministic report")
    second = builder.build(context(), findings, title="Deterministic report")
    assert first.report_id == second.report_id
    assert first.content_hash == second.content_hash
    markdown = render_markdown(first)
    assert "## Evidence appendix" in markdown
    assert EVIDENCE["checksum"] in markdown
    assert findings[0].statement in markdown
    changed = builder.build(
        context(), findings, title="Deterministic report",
        limitations=("A new limitation.",),
    )
    assert changed.report_id != first.report_id


def test_report_publication_requires_approval_and_confines_output(tmp_path):
    result = DiagnosticTool().invoke(ToolCall("call", "diagnose", {}))
    report = EngineeringReportBuilder().build(
        context(), DiagnosticSpecialist().analyze(result), title="Publish me",
    )
    catalog = ReportCatalog((tmp_path,))
    catalog.register(report)
    render_tool = RenderEngineeringReportTool(catalog)
    rendered = render_tool.invoke(ToolCall(
        "render", "render_engineering_report", {"report_id": report.report_id},
    ))
    assert rendered.value["content_hash"] == report.content_hash

    publish = PublishEngineeringReportTool(catalog)
    orchestrator = DeterministicOrchestrator((publish,))
    publish_context = context(
        AgentPermission.PUBLISH, allowed=("publish_engineering_report",),
    )
    publish_plan = InvestigationPlan((ToolCall(
        "publish", "publish_engineering_report",
        {"report_id": report.report_id, "path": str(tmp_path / "report.md")},
    ),))
    waiting = orchestrator.execute(publish_context, publish_plan)
    assert waiting.status is InvestigationStatus.WAITING_APPROVAL
    assert not (tmp_path / "report.md").exists()
    complete = orchestrator.execute(
        publish_context, publish_plan, approved_call_ids=("publish",),
    )
    assert complete.status is InvestigationStatus.COMPLETED
    assert (tmp_path / "report.md").read_text().startswith("# Publish me")

    with pytest.raises(ValueError, match="outside configured roots"):
        catalog.write_markdown(report.report_id, tmp_path.parent / "escape.md")


def test_investigation_bundle_roundtrips_and_detects_tampering(tmp_path):
    outcome = SingleCaseInvestigator(
        DeterministicOrchestrator((DiagnosticTool(),)), (DiagnosticSpecialist(),),
    ).investigate(
        context(), plan(), title="Portable review", qualification_passed=True,
    )
    target = write_investigation_bundle(tmp_path / "bundle", outcome)
    assert read_investigation_bundle(target) == outcome
    assert {item.name for item in target.iterdir()} == {
        "manifest.json", "outcome.json", "investigation.json",
        "report.json", "report.md",
    }
    with pytest.raises(FileExistsError):
        write_investigation_bundle(target, outcome)

    report_path = target / "report.md"
    report_path.write_text(report_path.read_text() + "tampered\n")
    with pytest.raises(ValueError, match="checksum"):
        read_investigation_bundle(target)


def test_failed_investigation_bundle_has_no_report_payload(tmp_path):
    outcome = SingleCaseInvestigator(
        DeterministicOrchestrator((DiagnosticTool(),)), (),
    ).investigate(context(), plan(), title="Failed review")
    target = write_investigation_bundle(tmp_path / "failed_bundle", outcome)
    restored = read_investigation_bundle(target)
    assert restored == outcome and restored.report is None
    assert not (target / "report.md").exists()
