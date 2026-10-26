"""Engineering acceptance tests for the complete Agent M1 boundary."""

import hashlib

import pytest

from pymrst.research_studio.agents import (
    AgentPermission,
    DeterministicOrchestrator,
    InvestigationContext,
    InvestigationPlan,
    InvestigationRecord,
    InvestigationStatus,
    PermissionGrant,
    Selection,
    ToolCall,
    ToolDescriptor,
    ToolResult,
)
from pymrst.research_studio.agents.policies.evidence import (
    EvidencePolicy,
    authorize_publication,
    qualify_finding,
)
from pymrst.research_studio.agents.policies.execution import (
    DecisionCode,
    ResourceEstimate,
    authorize_run,
)
from pymrst.research_studio.agents.policies.mutation import (
    CasePatchIntent,
    authorize_case_patch,
)
from pymrst.research_studio.agents.policies.resource_limits import ExecutionBudget
from pymrst.research_studio.agents.schemas.engineering_report import (
    EngineeringReport,
    ReportSection,
)
from pymrst.research_studio.agents.schemas.evidence import (
    EvidenceKind,
    EvidenceReference,
)
from pymrst.research_studio.agents.schemas.findings import (
    ClaimKind,
    EngineeringFinding,
    FindingSeverity,
)
from pymrst.research_studio.agents.schemas.run_request import RunRequest
from pymrst.research_studio.agents.schemas.run_result import (
    RunFailure,
    RunResult,
    RunStatus,
)


DIGEST = hashlib.sha256(b"case").hexdigest()


def context(permission=AgentPermission.ANALYZE):
    return InvestigationContext(
        "investigation_spe1",
        "Why did convergence slow at report step 57?",
        "case_spe1",
        DIGEST,
        PermissionGrant(permission, allowed_tools=("inspect", "run_case")),
        run_ids=("run_spe1",),
        selection=Selection(cells=(1, 2), wells=("PROD-2",), report_steps=(57,)),
        attributes={"ui": {"active_plot": "cnv", "layers": [1, 2]}},
    )


def reference(checksum="abc123"):
    return EvidenceReference(
        EvidenceKind.CONVERGENCE,
        "run://run_spe1/convergence",
        "Convergence history",
        {"report_step": 57},
        checksum=checksum,
    )


def finding(confidence=0.8, checksum="abc123"):
    return EngineeringFinding(
        "finding_1", "numerical_diagnostics", "convergence",
        FindingSeverity.WARNING, ClaimKind.INFERRED,
        "The control transition is associated with slower convergence.",
        (reference(checksum),), confidence,
        ("well:PROD-2", "report_step:57"),
    )


def run_request(budget=None):
    return RunRequest(
        "request_1", "run_new", "case_spe1", DIGEST, "local",
        {"owner": "blackoil", "options": {"phases": ["W", "O", "G"]}},
        {"nonlinear": {"max_iterations": 12}},
        ("pressure", "saturation", "well_solutions"),
        budget or ExecutionBudget(max_cells=1_000, max_report_steps=120),
        random_seed=42,
    )


def test_context_is_deeply_immutable_and_strictly_roundtrips():
    value = context()
    with pytest.raises(TypeError):
        value.attributes["ui"]["active_plot"] = "pressure"
    with pytest.raises(TypeError):
        value.attributes["ui"]["layers"][0] = 9
    assert InvestigationContext.from_dict(value.to_dict()) == value


def test_run_request_and_results_enforce_lifecycle_contracts():
    request = run_request()
    assert RunRequest.from_dict(request.to_dict()) == request
    complete = RunResult(
        request.request_id, request.run_id, RunStatus.COMPLETED,
        "store://runs/run_new", (reference(),), {"elapsed_seconds": 3.2},
    )
    assert RunResult.from_dict(complete.to_dict()) == complete
    with pytest.raises(ValueError, match="result_store_uri"):
        RunResult(request.request_id, request.run_id, RunStatus.COMPLETED)
    with pytest.raises(ValueError, match="failure details"):
        RunResult(request.request_id, request.run_id, RunStatus.FAILED)
    failed = RunResult(
        request.request_id, request.run_id, RunStatus.FAILED,
        failure=RunFailure("RuntimeError", "Newton failed", True),
    )
    assert failed.failure.retryable


def test_execution_policy_requires_approval_and_enforces_effective_budget():
    grant = PermissionGrant(
        AgentPermission.RUN_SIMULATION,
        ExecutionBudget(max_cells=500, max_report_steps=120),
        ("run_case",),
    )
    estimate = ResourceEstimate(cells=300, report_steps=120)
    assert authorize_run(run_request(), grant, estimate, approved=False).code \
        is DecisionCode.REQUIRE_APPROVAL
    assert authorize_run(run_request(), grant, estimate, approved=True).allowed
    denied = authorize_run(
        run_request(), grant, ResourceEstimate(cells=501, report_steps=120),
        approved=True,
    )
    assert denied.code is DecisionCode.DENY_RESOURCE_LIMIT


def test_mutation_policy_never_allows_original_case_writes():
    patch = CasePatchIntent(
        "patch_fault", DIGEST, "Test fault transmissibility uncertainty",
        ({"operation": "set_fault_multiplier", "fault": "F1", "value": 0.1},),
    )
    grant = PermissionGrant(AgentPermission.CREATE_VARIANT)
    assert authorize_case_patch(patch, grant, approved=True).allowed
    denied = authorize_case_patch(
        patch, grant, approved=True, writes_original=True,
    )
    assert denied.code is DecisionCode.DENY_MUTATION


def test_evidence_policy_blocks_weak_or_unverifiable_publication_claims():
    policy = EvidencePolicy(minimum_inferred_confidence=0.75)
    assert qualify_finding(finding(0.8), policy).allowed
    assert not qualify_finding(finding(0.7), policy).allowed
    assert not authorize_publication(
        (finding(),), qualified_run=False, policy=policy,
    ).allowed
    assert not authorize_publication(
        (finding(checksum=None),), qualified_run=True, policy=policy,
    ).allowed
    assert authorize_publication(
        (finding(),), qualified_run=True, policy=policy,
    ).allowed


def test_engineering_report_referential_integrity_and_roundtrip():
    item = finding()
    report = EngineeringReport(
        "report_1", "investigation_spe1", "SPE1 convergence review",
        "A control transition coincides with slower nonlinear convergence.",
        (item,),
        (ReportSection(
            "numerical_quality", "Numerical quality", "The run completed.",
            (item.finding_id,), (reference(),),
        ),),
        limitations=("Association does not establish causation.",),
        qualified=True,
    )
    assert EngineeringReport.from_dict(report.to_dict()) == report
    with pytest.raises(ValueError, match="unknown findings"):
        EngineeringReport(
            "report_2", "investigation_spe1", "Bad report", "Summary",
            (item,), (ReportSection("section", "Section", "Text", ("missing",)),),
        )


class InspectTool:
    descriptor = ToolDescriptor(
        "inspect", "Inspect an existing case", AgentPermission.INSPECT,
    )

    def invoke(self, call):
        return ToolResult(call.call_id, True, {"case": call.arguments["case"]})


class RunTool:
    descriptor = ToolDescriptor(
        "run_case", "Run a bounded case", AgentPermission.RUN_SIMULATION,
        starts_workload=True,
    )

    def invoke(self, call):
        return ToolResult(call.call_id, True, {"run_id": "run_new"})


class BrokenTool:
    descriptor = ToolDescriptor(
        "broken", "Raise a controlled test error", AgentPermission.INSPECT,
    )

    def invoke(self, call):
        raise RuntimeError("controlled tool failure")


def test_orchestrator_records_success_and_never_bypasses_approval():
    orchestrator = DeterministicOrchestrator((InspectTool(), RunTool()))
    inspect_plan = InvestigationPlan((
        ToolCall("call_inspect", "inspect", {"case": "case_spe1"}),
    ))
    complete = orchestrator.execute(context(), inspect_plan)
    assert complete.status is InvestigationStatus.COMPLETED
    assert len(complete.entries) == 2
    assert InvestigationRecord.from_json(complete.to_json()) == complete

    run_plan = InvestigationPlan((ToolCall("call_run", "run_case", {}),))
    waiting = orchestrator.execute(
        context(AgentPermission.RUN_SIMULATION), run_plan,
    )
    assert waiting.status is InvestigationStatus.WAITING_APPROVAL
    approved = orchestrator.execute(
        context(AgentPermission.RUN_SIMULATION), run_plan,
        approved_call_ids=("call_run",),
    )
    assert approved.status is InvestigationStatus.COMPLETED


def test_orchestrator_isolates_tool_exceptions_in_the_record():
    ctx = InvestigationContext(
        "investigation_failure", "Test isolation", "case_spe1", DIGEST,
        PermissionGrant(AgentPermission.INSPECT, allowed_tools=("broken",)),
    )
    record = DeterministicOrchestrator((BrokenTool(),)).execute(
        ctx, InvestigationPlan((ToolCall("call_broken", "broken", {}),)),
    )
    assert record.status is InvestigationStatus.FAILED
    result = record.entries[-1].payload
    assert result["error"] == "controlled tool failure"
    assert result["value"]["exception_type"] == "RuntimeError"
