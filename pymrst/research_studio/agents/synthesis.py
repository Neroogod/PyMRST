"""Deterministic cross-specialist synthesis with no language-model arithmetic."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Iterable, Mapping

from ._serialization import require_fields
from .review import (
    FindingConflict, HumanReviewDecision, UncertaintyAssessment,
    assess_uncertainty, detect_finding_conflicts,
)
from .schemas.findings import EngineeringFinding


@dataclass(frozen=True, slots=True)
class FindingCorroboration:
    corroboration_id: str
    finding_ids: tuple[str, ...]
    category: str
    shared_objects: tuple[str, ...]
    reason: str

    def __post_init__(self):
        identifiers = tuple(sorted(set(str(item).strip() for item in self.finding_ids)))
        if len(identifiers) < 2 or any(not item for item in identifiers):
            raise ValueError("corroboration requires at least two distinct findings")
        object.__setattr__(self, "finding_ids", identifiers)
        objects = tuple(sorted(set(str(item).strip() for item in self.shared_objects)))
        if not objects or any(not item for item in objects):
            raise ValueError("corroboration requires shared affected objects")
        object.__setattr__(self, "shared_objects", objects)
        for name in ("corroboration_id", "category", "reason"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)

    def to_dict(self):
        return {
            "corroboration_id": self.corroboration_id,
            "finding_ids": list(self.finding_ids), "category": self.category,
            "shared_objects": list(self.shared_objects), "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        data = require_fields(value, required={
            "corroboration_id", "finding_ids", "category", "shared_objects", "reason",
        }, name="FindingCorroboration")
        data["finding_ids"] = tuple(data["finding_ids"])
        data["shared_objects"] = tuple(data["shared_objects"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class FindingSynthesis:
    finding_ids: tuple[str, ...]
    conflicts: tuple[FindingConflict, ...]
    uncertainties: tuple[UncertaintyAssessment, ...]
    corroborations: tuple[FindingCorroboration, ...]
    summary: Mapping[str, Any]

    def __post_init__(self):
        identifiers = tuple(str(item).strip() for item in self.finding_ids)
        if len(identifiers) != len(set(identifiers)) or any(not item for item in identifiers):
            raise ValueError("synthesis finding ids must be unique and nonempty")
        object.__setattr__(self, "finding_ids", identifiers)
        object.__setattr__(self, "conflicts", tuple(self.conflicts))
        object.__setattr__(self, "uncertainties", tuple(self.uncertainties))
        object.__setattr__(self, "corroborations", tuple(self.corroborations))
        object.__setattr__(self, "summary", dict(self.summary))

    @property
    def unresolved_conflicts(self):
        return tuple(item for item in self.conflicts if not item.resolved)

    def to_dict(self):
        return {
            "schema_version": 1, "finding_ids": list(self.finding_ids),
            "conflicts": [item.to_dict() for item in self.conflicts],
            "uncertainties": [item.to_dict() for item in self.uncertainties],
            "corroborations": [item.to_dict() for item in self.corroborations],
            "summary": dict(self.summary),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]):
        data = require_fields(value, required={
            "schema_version", "finding_ids", "conflicts", "uncertainties",
            "corroborations", "summary",
        }, name="FindingSynthesis")
        if data.pop("schema_version") != 1:
            raise ValueError("unsupported FindingSynthesis schema version")
        data["finding_ids"] = tuple(data["finding_ids"])
        data["conflicts"] = tuple(FindingConflict.from_dict(item) for item in data["conflicts"])
        data["uncertainties"] = tuple(
            UncertaintyAssessment.from_dict(item) for item in data["uncertainties"]
        )
        data["corroborations"] = tuple(
            FindingCorroboration.from_dict(item) for item in data["corroborations"]
        )
        return cls(**data)


def _normalize_statement(value: str) -> str:
    return " ".join(str(value).casefold().split())


def synthesize_findings(
    findings: Iterable[EngineeringFinding],
    decisions: Iterable[HumanReviewDecision] = (),
) -> FindingSynthesis:
    findings = tuple(findings)
    decisions = tuple(decisions)
    identifiers = tuple(item.finding_id for item in findings)
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("synthesis findings must have unique ids")
    conflicts = detect_finding_conflicts(findings, decisions)
    uncertainties = assess_uncertainty(findings)
    corroborations = []
    for index, left in enumerate(findings):
        for right in findings[index + 1:]:
            shared = tuple(sorted(set(left.affected_objects) & set(right.affected_objects)))
            if (left.agent == right.agent or left.category != right.category or not shared
                    or _normalize_statement(left.statement) != _normalize_statement(right.statement)):
                continue
            pair = tuple(sorted((left.finding_id, right.finding_id)))
            identity = hashlib.sha256(
                "|".join((left.category, *shared, *pair)).encode()
            ).hexdigest()[:20]
            corroborations.append(FindingCorroboration(
                f"corroboration_{identity}", pair, left.category, shared,
                "Independent specialists made the same claim about the same affected object.",
            ))
    severity_counts = {}
    for item in findings:
        severity_counts[item.severity.value] = severity_counts.get(item.severity.value, 0) + 1
    summary = {
        "finding_count": len(findings),
        "agent_count": len({item.agent for item in findings}),
        "evidence_reference_count": sum(len(item.evidence) for item in findings),
        "severity_counts": severity_counts,
        "conflict_count": len(conflicts),
        "unresolved_conflict_count": sum(not item.resolved for item in conflicts),
        "uncertainty_count": len(uncertainties),
        "corroboration_count": len(corroborations),
    }
    return FindingSynthesis(
        identifiers, conflicts, uncertainties, tuple(corroborations), summary,
    )


__all__ = ["FindingCorroboration", "FindingSynthesis", "synthesize_findings"]
