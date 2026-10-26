"""Agent M1 contracts remain deterministic and simulation-free."""

from dataclasses import FrozenInstanceError

import pytest

from pymrst.research_studio.agents import (
    AgentPermission,
    ClaimKind,
    EngineeringFinding,
    EvidenceKind,
    EvidenceReference,
    ExecutionBudget,
    FindingSeverity,
    PermissionGrant,
    Recommendation,
    RecommendationStatus,
    ToolCall,
    ToolResult,
)


def _reference() -> EvidenceReference:
    return EvidenceReference(
        EvidenceKind.CONVERGENCE,
        "run://run_spe1/convergence",
        "SPE1 convergence at report step 57",
        {"report_step": 57, "newton_iteration": 3},
    )


def test_evidence_reference_is_immutable_and_json_safe():
    reference = _reference()
    with pytest.raises(TypeError):
        reference.locator["report_step"] = 58
    with pytest.raises(FrozenInstanceError):
        reference.label = "changed"
    assert EvidenceReference.from_dict(reference.to_dict()) == reference
    with pytest.raises(TypeError, match="unsupported persistent value"):
        EvidenceReference(
            EvidenceKind.STATE, "run://run_1/state", "state", {"model": object()}
        )


def test_engineering_finding_roundtrip_requires_evidence():
    recommendation = Recommendation(
        "rec_reduce_dt",
        "Create a timestep-sensitivity variant.",
        "The control transition and residual increase occur at the same step.",
    )
    finding = EngineeringFinding(
        "finding_control_transition",
        "numerical_diagnostics",
        "nonlinear_convergence",
        FindingSeverity.WARNING,
        ClaimKind.INFERRED,
        "A well-control transition is associated with slower convergence.",
        (_reference(),),
        0.82,
        ("well:PROD-2", "report_step:57"),
        recommendation,
    )
    assert EngineeringFinding.from_dict(finding.to_dict()) == finding
    with pytest.raises(ValueError, match="supporting evidence"):
        EngineeringFinding(
            "finding_unsupported", "agent", "category",
            FindingSeverity.INFO, ClaimKind.OBSERVED, "Unsupported claim", (), 0.5,
        )


def test_recommendation_state_cannot_claim_unconfirmed_application():
    with pytest.raises(ValueError, match="cannot still require confirmation"):
        Recommendation(
            "rec_1", "Change the model", "diagnostic test",
            requires_confirmation=True,
            status=RecommendationStatus.APPLIED,
        )


def test_permission_grant_is_explicit_and_budgeted():
    grant = PermissionGrant(
        AgentPermission.RUN_SIMULATION,
        ExecutionBudget(max_cells=300, max_report_steps=120),
        ("inspect_deck", "run_case"),
    )
    assert grant.permits(AgentPermission.INSPECT, "inspect_deck")
    assert grant.permits(AgentPermission.RUN_SIMULATION, "run_case")
    assert not grant.permits(AgentPermission.PUBLISH)
    assert not grant.permits(AgentPermission.ANALYZE, "unknown_tool")


def test_tool_boundary_rejects_runtime_objects_and_invalid_failures():
    call = ToolCall("call_1", "inspect_deck", {"case_id": "case_spe1"})
    assert call.arguments["case_id"] == "case_spe1"
    with pytest.raises(TypeError, match="unsupported persistent value"):
        ToolCall("call_2", "run_case", {"model": object()})
    with pytest.raises(ValueError, match="require an error"):
        ToolResult("call_3", False)
    with pytest.raises(ValueError, match="cannot contain an error"):
        ToolResult("call_4", True, error="not really successful")
