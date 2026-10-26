"""Acceptance tests for Agent M6 experiments, references, and report formats."""

from __future__ import annotations

import hashlib
import json

import numpy as np

from pymrst.research_studio.agents import (
    AgentPermission,
    AnalyzeEnsembleTool,
    CompareReferenceRunsTool,
    DeterministicOrchestrator,
    EnsembleAnalysisCatalog,
    ExperimentRequestCatalog,
    ExperimentSubmissionRequest,
    ExperimentSubmissionResult,
    InvestigationContext,
    InvestigationPlan,
    InvestigationStatus,
    PermissionGrant,
    ReferenceRunCatalog,
    SingleCaseInvestigator,
    SubmitExperimentTool,
    ToolCall,
)
from pymrst.research_studio.agents.policies.execution import ResourceEstimate
from pymrst.research_studio.agents.policies.resource_limits import ExecutionBudget
from pymrst.research_studio.agents.schemas.evidence import EvidenceReference
from pymrst.research_studio.agents.schemas.findings import (
    ClaimKind,
    EngineeringFinding,
    FindingSeverity,
)
from pymrst.research_studio.agents.specialists.reporting import ReportingSpecialist
from pymrst.research_studio.agents.specialists.simulator_disagreement import (
    SimulatorDisagreementSpecialist,
)
from pymrst.research_studio.agents.specialists.uncertainty import (
    UncertaintySpecialist,
)
from pymrst.research_studio.agents.tools.report_tools import (
    EngineeringReportBuilder,
    ReportCatalog,
    ReportCompositionTool,
    render_html,
)


CASE_HASH = hashlib.sha256(b"m6-case").hexdigest()


def context(permission=AgentPermission.ANALYZE, allowed=()):
    return InvestigationContext(
        "investigation_m6", "Compare scenarios and quantify uncertainty",
        "case_m6", CASE_HASH,
        PermissionGrant(
            permission,
            ExecutionBudget(
                max_cells=1_000, max_report_steps=100, max_realizations=20,
                max_workers=4, max_runtime_seconds=300,
                max_memory_bytes=2_000_000_000,
            ),
            allowed,
        ),
    )


class FakeExperimentAdapter:
    def __init__(self):
        self.requests = []

    def submit_experiment(self, request):
        self.requests.append(request)
        return ExperimentSubmissionResult(
            request.request_id, request.experiment_id, "queued",
            f"experiment://{request.experiment_id}/results",
        )


def test_experiment_submission_is_seeded_bounded_and_approval_gated():
    definition_hash = hashlib.sha256(b"experiment-definition").hexdigest()
    request = ExperimentSubmissionRequest(
        "experiment_request", "experiment_m6", "definition://experiment_m6",
        definition_hash, 10, 42,
        ExecutionBudget(
            max_cells=500, max_report_steps=50, max_realizations=10,
            max_workers=2, max_runtime_seconds=120,
            max_memory_bytes=1_000_000_000,
        ),
    )
    assert ExperimentSubmissionRequest.from_dict(request.to_dict()) == request
    submission = ExperimentSubmissionResult(
        request.request_id, request.experiment_id, "queued",
        "experiment://experiment_m6/results",
    )
    assert ExperimentSubmissionResult.from_dict(submission.to_dict()) == submission
    catalog = ExperimentRequestCatalog()
    catalog.register(request, ResourceEstimate(
        cells=500, report_steps=50, realizations=10, workers=2,
        runtime_seconds=100, memory_bytes=900_000_000,
    ))
    adapter = FakeExperimentAdapter()
    orchestrator = DeterministicOrchestrator((SubmitExperimentTool(catalog, adapter),))
    plan = InvestigationPlan((ToolCall(
        "submit_experiment", "submit_experiment",
        {"request_id": request.request_id},
    ),))
    run_context = context(
        AgentPermission.RUN_ENSEMBLE_OR_REMOTE, ("submit_experiment",),
    )
    waiting = orchestrator.execute(run_context, plan)
    assert waiting.status is InvestigationStatus.WAITING_APPROVAL
    assert not adapter.requests
    complete = orchestrator.execute(
        run_context, plan, approved_call_ids=("submit_experiment",),
    )
    assert complete.status is InvestigationStatus.COMPLETED
    assert adapter.requests == [request]


def test_ensemble_analysis_preserves_failed_members_and_quantifies_distribution():
    catalog = EnsembleAnalysisCatalog()
    dataset = catalog.register(
        "ensemble_m6", ("m1", "m2", "m3", "m4"),
        ("completed", "completed", "failed", "completed"),
        {"recovery_factor": np.array([[0.20], [0.25], [np.nan], [0.35]])},
    )
    outcome = SingleCaseInvestigator(
        DeterministicOrchestrator((AnalyzeEnsembleTool(catalog),)),
        (UncertaintySpecialist(),),
    ).investigate(
        context(AgentPermission.ANALYZE, ("analyze_ensemble",)),
        InvestigationPlan((ToolCall(
            "analyze", "analyze_ensemble", {
                "dataset_id": dataset.dataset_id, "quantity": "recovery_factor",
                "percentiles": (10, 50, 90), "maximum_failure_fraction": 0.1,
            },
        ),)), title="Ensemble uncertainty",
    )
    assert outcome.record.status is InvestigationStatus.COMPLETED
    assert len(outcome.findings) == 1
    assert "failure fraction" in outcome.findings[0].statement.lower()
    result = next(entry.payload["value"] for entry in outcome.record.entries
                  if entry.kind.value == "tool_result")
    assert result["summary"]["completed_members"] == 3
    assert result["summary"]["failed_members"] == 1
    assert np.isclose(result["summary"]["mean"][0], (0.2 + 0.25 + 0.35) / 3)


def _reference_spec(values, entities=("P1", "P2")):
    return {
        "values": values, "times": (1.0, 2.0), "entity_names": entities,
        "unit": "Pa", "sign_convention": "positive",
    }


def test_mrst_opm_or_pymrst_disagreement_uses_explicit_normalization():
    catalog = ReferenceRunCatalog()
    mrst = catalog.register("mrst_run", "mrst", {
        "bhp": _reference_spec(((10.0, 20.0), (11.0, 21.0))),
    })
    pymrst = catalog.register("pymrst_run", "pymrst", {
        # Reversed entity order is intentional and must be aligned by name.
        "bhp": _reference_spec(((20.0, 10.0), (21.5, 11.0)), ("P2", "P1")),
    })
    tool = CompareReferenceRunsTool(catalog)
    outcome = SingleCaseInvestigator(
        DeterministicOrchestrator((tool,)),
        (SimulatorDisagreementSpecialist(),),
    ).investigate(
        context(AgentPermission.ANALYZE, ("compare_reference_runs",)),
        InvestigationPlan((ToolCall(
            "compare", "compare_reference_runs", {
                "reference_run_id": mrst.run_id,
                "candidate_run_id": pymrst.run_id,
                "quantities": {"bhp": {
                    "absolute": 0.1, "relative": 0.0, "scale": 1.0,
                    "unit_scale": 1.0,
                }},
            },
        ),)), title="MRST and PyMRST comparison",
    )
    assert outcome.record.status is InvestigationStatus.COMPLETED
    assert len(outcome.findings) == 1
    assert "pymrst and mrst disagree" in outcome.findings[0].statement.lower()
    assert len(outcome.findings[0].evidence) == 2
    result = next(entry.payload["value"] for entry in outcome.record.entries
                  if entry.kind.value == "tool_result")
    assert result["comparisons"]["bhp"]["worst_entity"] == "P2"
    assert np.isclose(result["comparisons"]["bhp"]["maximum_absolute_error"], 0.5)


def test_reports_render_safe_html_and_structured_json(tmp_path):
    # Reuse a real M6 comparison finding rather than manufacture report prose.
    catalog = ReferenceRunCatalog()
    catalog.register("mrst", "mrst", {"p": _reference_spec(((1, 2), (2, 3)))})
    catalog.register("opm", "opm", {"p": _reference_spec(((1, 2), (2, 4)))})
    tool = CompareReferenceRunsTool(catalog)
    outcome = SingleCaseInvestigator(
        DeterministicOrchestrator((tool,)), (SimulatorDisagreementSpecialist(),),
    ).investigate(
        context(AgentPermission.ANALYZE, ("compare_reference_runs",)),
        InvestigationPlan((ToolCall("compare", "compare_reference_runs", {
            "reference_run_id": "mrst", "candidate_run_id": "opm",
            "quantities": {"p": {"absolute": 0.1, "scale": 1}},
        }),)), title="<MRST & OPM> qualification",
    )
    report = outcome.report
    html = render_html(report)
    assert "&lt;MRST &amp; OPM&gt;" in html
    assert "<script" not in html
    reports = ReportCatalog((tmp_path,))
    reports.register(report)
    html_path = reports.write_report(report.report_id, tmp_path / "report.html")
    json_path = reports.write_report(report.report_id, tmp_path / "report.json")
    assert html_path.read_text().startswith("<!doctype html>")
    assert json.loads(json_path.read_text())["report_id"] == report.report_id


def test_report_composition_flags_unqualified_report_missing_review_gaps(tmp_path):
    # Reuse the disagreement fixture: an unqualified report with one error
    # finding that carries neither a recommendation nor declared limitations.
    catalog = ReferenceRunCatalog()
    catalog.register("mrst", "mrst", {"p": _reference_spec(((1, 2), (2, 3)))})
    catalog.register("opm", "opm", {"p": _reference_spec(((1, 2), (2, 4)))})
    tool = CompareReferenceRunsTool(catalog)
    outcome = SingleCaseInvestigator(
        DeterministicOrchestrator((tool,)), (SimulatorDisagreementSpecialist(),),
    ).investigate(
        context(AgentPermission.ANALYZE, ("compare_reference_runs",)),
        InvestigationPlan((ToolCall("compare", "compare_reference_runs", {
            "reference_run_id": "mrst", "candidate_run_id": "opm",
            "quantities": {"p": {"absolute": 0.1, "scale": 1}},
        }),)), title="Disagreement report",
    )
    reports = ReportCatalog()
    reports.register(outcome.report)
    result = ReportCompositionTool(reports).invoke(ToolCall(
        "audit", "analyze_report_composition", {"report_id": outcome.report.report_id},
    ))
    categories = {item["category"] for item in result.value["findings"]}
    assert categories == {
        "report.missing_recommendation", "report.missing_limitations",
        "report.not_qualified",
    }
    assert result.value["severity_counts"] == {"error": 1}
    findings = ReportingSpecialist().analyze(result)
    assert {item.category for item in findings} == categories
    assert all(item.agent == "reporting" for item in findings)


def test_report_composition_is_clean_for_a_qualified_informational_report():
    evidence = EvidenceReference(
        "validation", "run://demo/pressure", "Demo pressure evidence", {}, "deadbeef",
    )
    finding = EngineeringFinding(
        finding_id="finding_demo", agent="reservoir_performance",
        category="performance.pressure", severity=FindingSeverity.INFO,
        claim_kind=ClaimKind.CALCULATED,
        statement="Mean reservoir pressure changed by 3e6 Pa.",
        evidence=(evidence,), confidence=1.0,
    )
    report = EngineeringReportBuilder().build(
        context(), (finding,), title="Clean report", qualified=True,
    )
    reports = ReportCatalog()
    reports.register(report)
    result = ReportCompositionTool(reports).invoke(ToolCall(
        "audit", "analyze_report_composition", {"report_id": report.report_id},
    ))
    assert result.value["findings"] == ()
    assert result.value["ok"] is True
    assert ReportingSpecialist().analyze(result) == ()


def test_report_composition_flags_unverifiable_evidence():
    evidence = EvidenceReference(
        "validation", "run://demo/pressure", "Demo pressure evidence", {}, None,
    )
    finding = EngineeringFinding(
        finding_id="finding_demo", agent="reservoir_performance",
        category="performance.pressure", severity=FindingSeverity.INFO,
        claim_kind=ClaimKind.CALCULATED,
        statement="Mean reservoir pressure changed by 3e6 Pa.",
        evidence=(evidence,), confidence=1.0,
    )
    report = EngineeringReportBuilder().build(
        context(), (finding,), title="Unverifiable report", qualified=True,
    )
    reports = ReportCatalog()
    reports.register(report)
    result = ReportCompositionTool(reports).invoke(ToolCall(
        "audit", "analyze_report_composition", {"report_id": report.report_id},
    ))
    categories = {item["category"] for item in result.value["findings"]}
    assert categories == {"report.unverifiable_evidence"}
