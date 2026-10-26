"""Typed conflict, uncertainty, and human-review contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
import hashlib
from typing import Any, Iterable, Mapping

from ._serialization import frozen_json_mapping, require_fields
from .schemas.findings import ClaimKind, EngineeringFinding


class ReviewTarget(StrEnum):
    FINDING = "finding"
    CONFLICT = "conflict"
    REPORT = "report"


class ReviewAction(StrEnum):
    APPROVE = "approve"
    REJECT = "reject"
    REQUEST_CHANGES = "request_changes"
    RESOLVE = "resolve"
    DISMISS = "dismiss"


@dataclass(frozen=True, slots=True)
class HumanReviewDecision:
    decision_id: str
    target: ReviewTarget
    target_id: str
    action: ReviewAction
    reviewer: str
    rationale: str
    selected_finding_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("decision_id", "target_id", "reviewer", "rationale"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "target", ReviewTarget(self.target))
        object.__setattr__(self, "action", ReviewAction(self.action))
        selected = tuple(str(item).strip() for item in self.selected_finding_ids)
        if any(not item for item in selected) or len(set(selected)) != len(selected):
            raise ValueError("selected finding ids must be unique and nonempty")
        if selected and not (
            self.target is ReviewTarget.CONFLICT
            and self.action is ReviewAction.RESOLVE
        ):
            raise ValueError("only conflict-resolution decisions select findings")
        if self.target is ReviewTarget.REPORT and self.action in {
            ReviewAction.RESOLVE, ReviewAction.DISMISS,
        }:
            raise ValueError("report reviews must approve, reject, or request changes")
        object.__setattr__(self, "selected_finding_ids", selected)
        object.__setattr__(self, "metadata", frozen_json_mapping(self.metadata, "$.metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id, "target": self.target.value,
            "target_id": self.target_id, "action": self.action.value,
            "reviewer": self.reviewer, "rationale": self.rationale,
            "selected_finding_ids": list(self.selected_finding_ids),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "HumanReviewDecision":
        data = require_fields(value, required={
            "decision_id", "target", "target_id", "action", "reviewer",
            "rationale", "selected_finding_ids", "metadata",
        }, name="HumanReviewDecision")
        data["selected_finding_ids"] = tuple(data["selected_finding_ids"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class FindingConflict:
    conflict_id: str
    finding_ids: tuple[str, str]
    category: str
    shared_objects: tuple[str, ...]
    reason: str
    resolution: HumanReviewDecision | None = None

    def __post_init__(self) -> None:
        values = tuple(str(item).strip() for item in self.finding_ids)
        if len(values) != 2 or any(not item for item in values) or values[0] == values[1]:
            raise ValueError("a conflict requires two distinct finding ids")
        object.__setattr__(self, "finding_ids", values)
        for name in ("conflict_id", "category", "reason"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        objects = tuple(sorted(set(str(item).strip() for item in self.shared_objects)))
        if any(not item for item in objects):
            raise ValueError("shared object identifiers cannot be empty")
        object.__setattr__(self, "shared_objects", objects)
        if self.resolution is not None:
            decision = self.resolution
            if decision.target is not ReviewTarget.CONFLICT:
                raise ValueError("conflict resolution must target a conflict")
            if decision.target_id != self.conflict_id:
                raise ValueError("conflict resolution target does not match conflict")
            if decision.action not in {ReviewAction.RESOLVE, ReviewAction.DISMISS}:
                raise ValueError("conflict remains unresolved by this review action")
            if not set(decision.selected_finding_ids).issubset(self.finding_ids):
                raise ValueError("resolution selected a finding outside the conflict")
            if decision.action is ReviewAction.RESOLVE and not decision.selected_finding_ids:
                raise ValueError("resolved conflicts must select at least one finding")

    @property
    def resolved(self) -> bool:
        return self.resolution is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "conflict_id": self.conflict_id,
            "finding_ids": list(self.finding_ids), "category": self.category,
            "shared_objects": list(self.shared_objects), "reason": self.reason,
            "resolution": None if self.resolution is None else self.resolution.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FindingConflict":
        data = require_fields(value, required={
            "conflict_id", "finding_ids", "category", "shared_objects",
            "reason", "resolution",
        }, name="FindingConflict")
        data["finding_ids"] = tuple(data["finding_ids"])
        data["shared_objects"] = tuple(data["shared_objects"])
        if data["resolution"] is not None:
            data["resolution"] = HumanReviewDecision.from_dict(data["resolution"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class UncertaintyAssessment:
    finding_id: str
    uncertainty_kind: str
    confidence: float
    explanation: str

    def __post_init__(self) -> None:
        for name in ("finding_id", "uncertainty_kind", "explanation"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        confidence = float(self.confidence)
        if not 0 <= confidence <= 1:
            raise ValueError("uncertainty confidence must be between zero and one")
        object.__setattr__(self, "confidence", confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_id": self.finding_id,
            "uncertainty_kind": self.uncertainty_kind,
            "confidence": self.confidence, "explanation": self.explanation,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "UncertaintyAssessment":
        return cls(**require_fields(value, required={
            "finding_id", "uncertainty_kind", "confidence", "explanation",
        }, name="UncertaintyAssessment"))


def detect_finding_conflicts(
    findings: Iterable[EngineeringFinding],
    decisions: Iterable[HumanReviewDecision] = (),
) -> tuple[FindingConflict, ...]:
    """Flag divergent cross-specialist claims about the same category/object."""
    findings = tuple(findings)
    resolution_by_target = {
        item.target_id: item for item in decisions
        if item.target is ReviewTarget.CONFLICT
        and item.action in {ReviewAction.RESOLVE, ReviewAction.DISMISS}
    }
    output = []
    for index, left in enumerate(findings):
        for right in findings[index + 1:]:
            shared = tuple(sorted(set(left.affected_objects) & set(right.affected_objects)))
            if (
                left.agent == right.agent or left.category != right.category
                or not shared or left.statement == right.statement
            ):
                continue
            pair = tuple(sorted((left.finding_id, right.finding_id)))
            identity = hashlib.sha256(
                "|".join((left.category, *shared, *pair)).encode()
            ).hexdigest()[:20]
            conflict_id = f"conflict_{identity}"
            output.append(FindingConflict(
                conflict_id, pair, left.category, shared,
                "Different specialists made divergent claims about the same "
                "engineering category and affected object.",
                resolution_by_target.get(conflict_id),
            ))
    return tuple(output)


def assess_uncertainty(
    findings: Iterable[EngineeringFinding],
) -> tuple[UncertaintyAssessment, ...]:
    output = []
    for finding in findings:
        if finding.claim_kind is ClaimKind.UNCERTAIN:
            kind = "explicit"
        elif finding.claim_kind is ClaimKind.INFERRED:
            kind = "epistemic"
        elif finding.confidence < 1.0:
            kind = "measurement_or_numerical"
        else:
            continue
        output.append(UncertaintyAssessment(
            finding.finding_id, kind, finding.confidence,
            f"The {finding.claim_kind.value} claim is carried with confidence "
            f"{finding.confidence:.3f}; the investigation does not treat it as exact.",
        ))
    return tuple(output)


__all__ = [
    "FindingConflict", "HumanReviewDecision", "ReviewAction", "ReviewTarget",
    "UncertaintyAssessment", "assess_uncertainty", "detect_finding_conflicts",
]
