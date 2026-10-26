"""Application contracts for evidence-controlled engineering investigations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Mapping

from ._common import json_value, nonempty


class InvestigationStatus(StrEnum):
    OPEN = "open"
    COMPLETED = "completed"
    UNQUALIFIED = "unqualified"
    FAILED = "failed"


class FindingReviewStatus(StrEnum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


class VariantProposalStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass(frozen=True, slots=True)
class InvestigationRequest:
    question: str
    run_ids: tuple[str, ...] = ()
    selection: Mapping[str, Any] = field(default_factory=dict)
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "question", nonempty(self.question, "question"))
        run_ids = tuple(nonempty(item, "run id") for item in self.run_ids)
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("investigation run ids must be unique")
        object.__setattr__(self, "run_ids", run_ids)
        object.__setattr__(self, "selection", json_value(self.selection))
        object.__setattr__(self, "attributes", json_value(self.attributes))

    @classmethod
    def from_dict(cls, value):
        return cls(
            value["question"], tuple(value.get("run_ids", ())),
            value.get("selection", {}), value.get("attributes", {}),
        )


@dataclass(frozen=True, slots=True)
class InvestigationRecord:
    investigation_id: str
    project_id: str
    question: str
    status: InvestigationStatus
    context: Mapping[str, Any]
    content_hash: str
    created_at: str
    updated_at: str

    def to_dict(self):
        return {
            "investigation_id": self.investigation_id,
            "project_id": self.project_id, "question": self.question,
            "status": self.status.value, "context": dict(self.context),
            "content_hash": self.content_hash, "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class FindingRecord:
    finding_id: str
    investigation_id: str
    project_id: str
    finding: Mapping[str, Any]
    content_hash: str
    review_status: FindingReviewStatus
    review_note: str
    created_at: str
    updated_at: str

    def to_dict(self):
        return {
            "finding_id": self.finding_id,
            "investigation_id": self.investigation_id,
            "project_id": self.project_id, "finding": dict(self.finding),
            "content_hash": self.content_hash,
            "review_status": self.review_status.value,
            "review_note": self.review_note, "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class EngineeringReportRecord:
    report_id: str
    investigation_id: str
    project_id: str
    content_hash: str
    report: Mapping[str, Any]
    relative_path: str
    status: str
    created_at: str

    def to_dict(self):
        return {
            "report_id": self.report_id,
            "investigation_id": self.investigation_id,
            "project_id": self.project_id, "content_hash": self.content_hash,
            "report": dict(self.report), "relative_path": self.relative_path,
            "status": self.status, "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class VariantProposalRecord:
    proposal_id: str
    investigation_id: str
    finding_id: str
    project_id: str
    name: str
    proposal_kind: str
    content_hash: str
    proposal: Mapping[str, Any]
    status: VariantProposalStatus
    decision_note: str
    created_at: str
    updated_at: str

    def to_dict(self):
        return {
            "proposal_id": self.proposal_id,
            "investigation_id": self.investigation_id,
            "finding_id": self.finding_id, "project_id": self.project_id,
            "name": self.name, "proposal_kind": self.proposal_kind,
            "content_hash": self.content_hash, "proposal": dict(self.proposal),
            "status": self.status.value, "decision_note": self.decision_note,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class InvestigationPlanRecord:
    plan_id: str
    investigation_id: str
    project_id: str
    provider: str
    content_hash: str
    plan: Mapping[str, Any]
    created_at: str

    def to_dict(self):
        return {
            "plan_id": self.plan_id, "investigation_id": self.investigation_id,
            "project_id": self.project_id, "provider": self.provider,
            "content_hash": self.content_hash, "plan": dict(self.plan),
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class InvestigationExecutionRecord:
    execution_id: str
    investigation_id: str
    plan_id: str
    project_id: str
    plan_hash: str
    content_hash: str
    status: str
    execution: Mapping[str, Any]
    created_at: str
    completed_at: str

    def to_dict(self):
        return {
            "execution_id": self.execution_id,
            "investigation_id": self.investigation_id,
            "plan_id": self.plan_id, "project_id": self.project_id,
            "plan_hash": self.plan_hash, "content_hash": self.content_hash,
            "status": self.status, "execution": dict(self.execution),
            "created_at": self.created_at, "completed_at": self.completed_at,
        }


@dataclass(frozen=True, slots=True)
class InvestigationSynthesisRecord:
    synthesis_id: str
    investigation_id: str
    project_id: str
    finding_set_hash: str
    decision_set_hash: str
    content_hash: str
    synthesis: Mapping[str, Any]
    created_at: str

    def to_dict(self):
        return {
            "synthesis_id": self.synthesis_id,
            "investigation_id": self.investigation_id,
            "project_id": self.project_id,
            "finding_set_hash": self.finding_set_hash,
            "decision_set_hash": self.decision_set_hash,
            "content_hash": self.content_hash,
            "synthesis": dict(self.synthesis), "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class InvestigationNarrativeRecord:
    narrative_id: str
    investigation_id: str
    project_id: str
    provider: str
    provider_configuration_hash: str
    source_findings_hash: str
    content_hash: str
    narrative: Mapping[str, Any]
    status: str
    reviewer: str
    review_note: str
    created_at: str
    updated_at: str

    def to_dict(self):
        return {
            "narrative_id": self.narrative_id,
            "investigation_id": self.investigation_id,
            "project_id": self.project_id, "provider": self.provider,
            "provider_configuration_hash": self.provider_configuration_hash,
            "source_findings_hash": self.source_findings_hash,
            "content_hash": self.content_hash, "narrative": dict(self.narrative),
            "status": self.status, "reviewer": self.reviewer,
            "review_note": self.review_note, "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class ReasoningProviderRecord:
    provider_id: str
    kind: str
    model: str
    endpoint: str
    credential_env: str
    enabled: bool
    content_hash: str
    configuration: Mapping[str, Any]
    created_at: str
    updated_at: str

    def to_dict(self):
        return {
            "provider_id": self.provider_id, "kind": self.kind,
            "model": self.model, "endpoint": self.endpoint,
            "credential_env": self.credential_env, "enabled": self.enabled,
            "content_hash": self.content_hash,
            "configuration": dict(self.configuration),
            "created_at": self.created_at, "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class ReasoningProviderCheckRecord:
    check_id: str
    provider_id: str
    status: str
    latency_seconds: float
    configuration_hash: str
    details: Mapping[str, Any]
    created_at: str

    def to_dict(self):
        return {
            "check_id": self.check_id, "provider_id": self.provider_id,
            "status": self.status, "latency_seconds": self.latency_seconds,
            "configuration_hash": self.configuration_hash,
            "details": dict(self.details), "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class ReasoningProviderEvaluationRecord:
    evaluation_id: str
    provider_id: str
    configuration_hash: str
    suite_hash: str
    status: str
    content_hash: str
    report: Mapping[str, Any]
    created_at: str

    def to_dict(self):
        return {
            "evaluation_id": self.evaluation_id, "provider_id": self.provider_id,
            "configuration_hash": self.configuration_hash,
            "suite_hash": self.suite_hash, "status": self.status,
            "content_hash": self.content_hash, "report": dict(self.report),
            "created_at": self.created_at,
        }


__all__ = [
    "EngineeringReportRecord", "FindingRecord", "FindingReviewStatus",
    "InvestigationRecord", "InvestigationRequest", "InvestigationStatus",
    "VariantProposalRecord", "VariantProposalStatus",
    "InvestigationPlanRecord", "InvestigationExecutionRecord",
    "InvestigationSynthesisRecord",
    "InvestigationNarrativeRecord",
    "ReasoningProviderRecord",
    "ReasoningProviderCheckRecord",
    "ReasoningProviderEvaluationRecord",
]
