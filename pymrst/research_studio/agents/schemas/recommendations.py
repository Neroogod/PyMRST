"""Human-reviewable actions proposed by engineering investigations."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from .._serialization import require_fields


class RecommendationStatus(StrEnum):
    PROPOSED = "proposed"
    APPROVED = "approved"
    REJECTED = "rejected"
    APPLIED = "applied"


@dataclass(frozen=True, slots=True)
class Recommendation:
    recommendation_id: str
    statement: str
    rationale: str
    requires_confirmation: bool = True
    status: RecommendationStatus = RecommendationStatus.PROPOSED

    def __post_init__(self) -> None:
        for name in ("recommendation_id", "statement", "rationale"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "status", RecommendationStatus(self.status))
        if self.status is RecommendationStatus.APPLIED and self.requires_confirmation:
            raise ValueError(
                "an applied recommendation cannot still require confirmation"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recommendation_id": self.recommendation_id,
            "statement": self.statement,
            "rationale": self.rationale,
            "requires_confirmation": self.requires_confirmation,
            "status": self.status.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Recommendation":
        return cls(**require_fields(
            value,
            required={
                "recommendation_id", "statement", "rationale",
                "requires_confirmation", "status",
            },
            name="Recommendation",
        ))


__all__ = ["Recommendation", "RecommendationStatus"]
