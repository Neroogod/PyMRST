"""Acceptance tests for immutable variants and controlled agent execution."""

from __future__ import annotations

import hashlib

import pytest

from pymrst.research_studio.agents import (
    AgentPermission,
    CancelSimulationTool,
    CaseVariantCatalog,
    CaseVariantManifest,
    CheckpointLineage,
    CheckpointLineageCatalog,
    DiagnoseRunFailureTool,
    DeterministicOrchestrator,
    FailureClass,
    InspectSimulationTool,
    InvestigationContext,
    InvestigationPlan,
    InvestigationStatus,
    PermissionGrant,
    RetryProposal,
    RunRequestCatalog,
    RunResultCatalog,
    SingleCaseInvestigator,
    SubmitSimulationTool,
    ToolCall,
    diagnose_run_failure,
    materialize_resume_request,
    materialize_retry_request,
    register_case_variant,
)
from pymrst.research_studio.agents.policies.execution import ResourceEstimate
from pymrst.research_studio.agents.policies.mutation import CasePatchIntent
from pymrst.research_studio.agents.policies.resource_limits import ExecutionBudget
from pymrst.research_studio.agents.schemas.run_request import RunRequest
from pymrst.research_studio.agents.schemas.run_result import (
    RunFailure,
    RunResult,
    RunStatus,
)
from pymrst.research_studio.agents.specialists.failure_diagnosis import (
    FailureDiagnosisSpecialist,
)


CASE_HASH = hashlib.sha256(b"m5-case").hexdigest()
DIGESTS = [hashlib.sha256(f"fingerprint-{index}".encode()).hexdigest()
           for index in range(4)]


def request(*, request_id="request_m5", run_id="run_m5"):
    return RunRequest(
        request_id, run_id, "case_m5", CASE_HASH, "local",
        {"owner": "blackoil"}, {"max_iterations": 20},
        ("pressure", "well_solutions"),
        ExecutionBudget(max_cells=200, max_report_steps=20,
                        max_runtime_seconds=60, max_memory_bytes=1_000_000),
        random_seed=42,
    )


def context(maximum=AgentPermission.RUN_SIMULATION, allowed=("submit_simulation",)):
    return InvestigationContext(
        "investigation_m5", "Execute a bounded diagnostic run", "case_m5",
        CASE_HASH,
        PermissionGrant(
            maximum,
            ExecutionBudget(max_cells=100, max_report_steps=10,
                            max_runtime_seconds=30, max_memory_bytes=500_000),
            allowed,
        ),
    )


class FakeAdapter:
    def __init__(self):
        self.submissions = []
        self.inspections = []
        self.cancellations = []

    def submit(self, value):
        self.submissions.append(value)
        return RunResult(
            value.request_id, value.run_id, RunStatus.COMPLETED,
            f"store://runs/{value.run_id}", metrics={"elapsed_seconds": 0.01},
        )

    def inspect(self, run_id):
        self.inspections.append(run_id)
        return RunResult("request_m5", run_id, RunStatus.RUNNING)

    def cancel(self, run_id):
        self.cancellations.append(run_id)
        return RunResult("request_m5", run_id, RunStatus.CANCELLED)


def test_variant_manifest_is_content_addressed_append_only_and_roundtrips():
    patch = CasePatchIntent(
        "patch_fault", CASE_HASH, "Test fault uncertainty",
        ({"operation": "set_fault_multiplier", "fault": "F1", "value": 0.2},),
    )
    first = CaseVariantManifest.create("variant_f1", "case_m5", patch)
    second = CaseVariantManifest.create("variant_f1_copy", "case_m5", patch)
    assert first.content_hash == second.content_hash
    assert first.content_hash != CASE_HASH
    assert CaseVariantManifest.from_dict(first.to_dict()) == first
    catalog = CaseVariantCatalog()
    catalog.register(first)
    with pytest.raises(KeyError, match="already registered"):
        catalog.register(first)
    tampered = first.to_dict()
    tampered["content_hash"] = CASE_HASH
    with pytest.raises(ValueError, match="does not match"):
        CaseVariantManifest.from_dict(tampered)


def test_variant_registration_requires_create_variant_authority_and_approval():
    patch = CasePatchIntent(
        "patch_poro", CASE_HASH, "Test porosity uncertainty",
        ({"operation": "scale_porosity", "factor": 1.05},),
    )
    catalog = CaseVariantCatalog()
    with pytest.raises(PermissionError, match="create_variant"):
        register_case_variant(
            catalog, variant_id="variant_poro", base_case_id="case_m5",
            patch=patch, grant=PermissionGrant(AgentPermission.ANALYZE), approved=True,
        )
    with pytest.raises(PermissionError, match="approval"):
        register_case_variant(
            catalog, variant_id="variant_poro", base_case_id="case_m5",
            patch=patch, grant=PermissionGrant(AgentPermission.CREATE_VARIANT),
            approved=False,
        )
    manifest = register_case_variant(
        catalog, variant_id="variant_poro", base_case_id="case_m5",
        patch=patch, grant=PermissionGrant(AgentPermission.CREATE_VARIANT),
        approved=True,
    )
    assert manifest.base_case_hash == CASE_HASH
    assert catalog.get("variant_poro") == manifest


def test_submission_requires_approval_and_enforces_effective_resource_budget():
    adapter = FakeAdapter()
    catalog = RunRequestCatalog()
    catalog.register(request(), ResourceEstimate(
        cells=80, report_steps=8, runtime_seconds=20, memory_bytes=400_000,
    ))
    orchestrator = DeterministicOrchestrator((SubmitSimulationTool(catalog, adapter),))
    plan = InvestigationPlan((ToolCall(
        "submit", "submit_simulation", {"request_id": "request_m5"},
    ),))
    waiting = orchestrator.execute(context(), plan)
    assert waiting.status is InvestigationStatus.WAITING_APPROVAL
    assert not adapter.submissions
    complete = orchestrator.execute(context(), plan, approved_call_ids=("submit",))
    assert complete.status is InvestigationStatus.COMPLETED
    assert len(adapter.submissions) == 1

    too_large_catalog = RunRequestCatalog()
    too_large_catalog.register(request(request_id="large", run_id="large"), ResourceEstimate(
        cells=101, report_steps=8, runtime_seconds=20, memory_bytes=400_000,
    ))
    denied_adapter = FakeAdapter()
    denied = DeterministicOrchestrator((
        SubmitSimulationTool(too_large_catalog, denied_adapter),
    )).execute(
        context(), InvestigationPlan((ToolCall(
            "large_submit", "submit_simulation", {"request_id": "large"},
        ),)), approved_call_ids=("large_submit",),
    )
    assert denied.status is InvestigationStatus.FAILED
    assert "effective limit" in denied.entries[-1].payload["error"]
    assert not denied_adapter.submissions


def test_monitoring_is_read_only_but_cancellation_requires_approval():
    adapter = FakeAdapter()
    inspect_plan = InvestigationPlan((ToolCall(
        "inspect", "inspect_simulation", {"run_id": "run_m5"},
    ),))
    inspected = DeterministicOrchestrator((InspectSimulationTool(adapter),)).execute(
        context(AgentPermission.INSPECT, ("inspect_simulation",)), inspect_plan,
    )
    assert inspected.status is InvestigationStatus.COMPLETED
    assert adapter.inspections == ["run_m5"]

    cancel_plan = InvestigationPlan((ToolCall(
        "cancel", "cancel_simulation", {"run_id": "run_m5"},
    ),))
    orchestrator = DeterministicOrchestrator((CancelSimulationTool(adapter),))
    cancel_context = context(AgentPermission.RUN_SIMULATION, ("cancel_simulation",))
    waiting = orchestrator.execute(cancel_context, cancel_plan)
    assert waiting.status is InvestigationStatus.WAITING_APPROVAL
    assert not adapter.cancellations
    complete = orchestrator.execute(
        cancel_context, cancel_plan, approved_call_ids=("cancel",),
    )
    assert complete.status is InvestigationStatus.COMPLETED
    assert adapter.cancellations == ["run_m5"]


@pytest.mark.parametrize(("message", "expected"), (
    ("Newton failed to converge", FailureClass.NONLINEAR_CONVERGENCE),
    ("GMRES linear solver failed", FailureClass.LINEAR_SOLVER),
    ("NaN detected in residual", FailureClass.NONFINITE),
    ("memory resource exceeded", FailureClass.RESOURCE_LIMIT),
    ("unsupported deck keyword", FailureClass.INPUT_OR_CONFIGURATION),
))
def test_failure_diagnosis_is_deterministic(message, expected):
    result = RunResult(
        "request_m5", "run_m5", RunStatus.FAILED,
        failure=RunFailure("RuntimeError", message, retryable=True,
                           failure_bundle_uri="failure://run_m5/bundle"),
    )
    diagnosis = diagnose_run_failure(result)
    assert diagnosis.failure_class is expected
    assert diagnosis.diagnostic_actions


def test_retry_is_a_bounded_reviewed_new_request_and_never_mutates_source():
    source = request()
    failure = RunResult(
        source.request_id, source.run_id, RunStatus.FAILED,
        failure=RunFailure("ConvergenceError", "Newton did not converge", True),
    )
    proposal = RetryProposal(
        "retry_1", source.request_id, str(source.run_id), 1,
        "Replay from checkpoint with a smaller initial timestep.",
        diagnose_run_failure(failure),
        {"initial_timestep_factor": 0.5, "max_timestep_cuts": 8},
        "checkpoint_4",
    )
    grant = PermissionGrant(AgentPermission.RUN_SIMULATION)
    with pytest.raises(PermissionError, match="approval"):
        materialize_retry_request(
            source, proposal, new_request_id="retry_request",
            new_run_id="retry_run", grant=grant, approved=False,
        )
    retry = materialize_retry_request(
        source, proposal, new_request_id="retry_request",
        new_run_id="retry_run", grant=grant, approved=True,
    )
    assert source.solver == {"max_iterations": 20}
    assert retry.solver["initial_timestep_factor"] == 0.5
    assert retry.checkpoint_id == "checkpoint_4"
    assert retry.request_id != source.request_id and retry.run_id != source.run_id

    unsafe = RetryProposal(
        "retry_unsafe", source.request_id, str(source.run_id), 1,
        "Relax all tolerances.", diagnose_run_failure(failure),
        {"unknown_tolerance": 1000},
    )
    with pytest.raises(PermissionError, match="unsupported solver overrides"):
        materialize_retry_request(
            source, unsafe, new_request_id="unsafe", new_run_id="unsafe",
            grant=grant, approved=True,
        )


def test_checkpoint_resume_preserves_explicit_lineage():
    source = request()
    lineage = CheckpointLineage(
        "checkpoint_4", source.request_id, str(source.run_id), CASE_HASH, 4,
        *DIGESTS, failure_bundle_uri="failure://run_m5/bundle",
    )
    assert CheckpointLineage.from_dict(lineage.to_dict()) == lineage
    resumed = materialize_resume_request(
        source, lineage, new_request_id="resume_request", new_run_id="resume_run",
    )
    assert resumed.checkpoint_id == lineage.checkpoint_id
    assert resumed.case_content_hash == source.case_content_hash
    wrong = CheckpointLineage(
        "checkpoint_4", source.request_id, "another_run", CASE_HASH, 4, *DIGESTS,
    )
    with pytest.raises(ValueError, match="lineage"):
        materialize_resume_request(
            source, wrong, new_request_id="bad", new_run_id="bad",
        )


def test_run_status_history_and_failure_diagnosis_join_the_agent_pipeline():
    results = RunResultCatalog()
    results.record(RunResult("request_m5", "run_m5", RunStatus.QUEUED))
    results.record(RunResult("request_m5", "run_m5", RunStatus.RUNNING))
    failed = RunResult(
        "request_m5", "run_m5", RunStatus.FAILED,
        failure=RunFailure(
            "ConvergenceError", "Newton did not converge", True,
            "failure://run_m5/bundle",
        ),
    )
    results.record(failed)
    assert tuple(item.status for item in results.history("run_m5")) == (
        RunStatus.QUEUED, RunStatus.RUNNING, RunStatus.FAILED,
    )
    with pytest.raises(ValueError, match="terminal"):
        results.record(RunResult("request_m5", "run_m5", RunStatus.CANCELLED))

    checkpoints = CheckpointLineageCatalog()
    lineage = checkpoints.register(CheckpointLineage(
        "checkpoint_4", "request_m5", "run_m5", CASE_HASH, 4, *DIGESTS,
        failure_bundle_uri="failure://run_m5/bundle",
    ))
    tool = DiagnoseRunFailureTool(results, checkpoints)
    diagnosis_context = context(
        AgentPermission.ANALYZE, ("diagnose_run_failure",),
    )
    outcome = SingleCaseInvestigator(
        DeterministicOrchestrator((tool,)), (FailureDiagnosisSpecialist(),),
    ).investigate(
        diagnosis_context,
        InvestigationPlan((ToolCall(
            "diagnose", "diagnose_run_failure", {"run_id": "run_m5"},
        ),)),
        title="Failure investigation",
    )
    assert outcome.record.status is InvestigationStatus.COMPLETED
    assert outcome.report is not None
    assert "nonlinear" in outcome.findings[0].category
    tool_result = next(
        entry for entry in outcome.record.entries if entry.kind.value == "tool_result"
    )
    assert tool_result.payload["value"]["checkpoint_lineage"][0]["checkpoint_id"] \
        == lineage.checkpoint_id


def test_retry_proposal_is_persistent_and_strict():
    failed = RunResult(
        "request_m5", "run_m5", RunStatus.FAILED,
        failure=RunFailure("ConvergenceError", "Newton failed", True),
    )
    proposal = RetryProposal(
        "retry_persistent", "request_m5", "run_m5", 2,
        "Use the accepted checkpoint and reduce the first timestep.",
        diagnose_run_failure(failed), {"initial_timestep_factor": 0.25},
        "checkpoint_4",
    )
    assert RetryProposal.from_dict(proposal.to_dict()) == proposal
