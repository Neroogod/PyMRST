"""Typed engineering claims and human-reviewable recommendations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from .._serialization import json_value, require_fields
from .evidence import EvidenceReference
from .recommendations import Recommendation, RecommendationStatus


AGENT_FINDING_SCHEMA_VERSION = 1


class FindingSeverity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ClaimKind(StrEnum):
    OBSERVED = "observed"
    CALCULATED = "calculated"
    INFERRED = "inferred"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True, slots=True)
class EngineeringFinding:
    finding_id: str
    agent: str
    category: str
    severity: FindingSeverity
    claim_kind: ClaimKind
    statement: str
    evidence: tuple[EvidenceReference, ...]
    confidence: float
    affected_objects: tuple[str, ...] = ()
    recommendation: Recommendation | None = None
    schema_version: int = AGENT_FINDING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != AGENT_FINDING_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported EngineeringFinding schema version "
                f"{self.schema_version!r}"
            )
        for name in ("finding_id", "agent", "category", "statement"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "severity", FindingSeverity(self.severity))
        object.__setattr__(self, "claim_kind", ClaimKind(self.claim_kind))
        evidence = tuple(
            item if isinstance(item, EvidenceReference)
            else EvidenceReference.from_dict(item)
            for item in self.evidence
        )
        if not evidence:
            raise ValueError("engineering findings require supporting evidence")
        object.__setattr__(self, "evidence", evidence)
        confidence = float(self.confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")
        object.__setattr__(self, "confidence", confidence)
        objects = tuple(str(item).strip() for item in self.affected_objects)
        if any(not item for item in objects):
            raise ValueError("affected object identifiers cannot be empty")
        object.__setattr__(self, "affected_objects", objects)
        if self.recommendation is not None and not isinstance(
            self.recommendation, Recommendation
        ):
            object.__setattr__(
                self,
                "recommendation",
                Recommendation.from_dict(self.recommendation),
            )

    def to_dict(self) -> dict[str, Any]:
        return json_value({
            "schema_version": self.schema_version,
            "finding_id": self.finding_id,
            "agent": self.agent,
            "category": self.category,
            "severity": self.severity.value,
            "claim_kind": self.claim_kind.value,
            "statement": self.statement,
            "evidence": [item.to_dict() for item in self.evidence],
            "confidence": self.confidence,
            "affected_objects": list(self.affected_objects),
            "recommendation": (
                None if self.recommendation is None
                else self.recommendation.to_dict()
            ),
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EngineeringFinding":
        data = require_fields(
            value,
            required={
                "schema_version", "finding_id", "agent", "category",
                "severity", "claim_kind", "statement", "evidence",
                "confidence", "affected_objects", "recommendation",
            },
            name="EngineeringFinding",
        )
        data["evidence"] = tuple(
            EvidenceReference.from_dict(item) for item in data["evidence"]
        )
        if data["recommendation"] is not None:
            data["recommendation"] = Recommendation.from_dict(
                data["recommendation"]
            )
        return cls(**data)


__all__ = [
    "AGENT_FINDING_SCHEMA_VERSION",
    "ClaimKind",
    "EngineeringFinding",
    "FindingSeverity",
    "Recommendation",
    "RecommendationStatus",
]
