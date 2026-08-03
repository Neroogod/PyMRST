"""Stable, JSON-safe identifiers used by persisted Research Studio records."""

from __future__ import annotations

from typing import Self
from uuid import UUID, uuid4


class StableId(str):
    """Non-empty opaque identifier with a type-specific generated prefix."""

    prefix = "id"

    def __new__(cls, value: str) -> Self:
        value = str(value).strip()
        if not value:
            raise ValueError(f"{cls.__name__} cannot be empty")
        if any(character.isspace() for character in value):
            raise ValueError(f"{cls.__name__} cannot contain whitespace")
        return str.__new__(cls, value)

    @classmethod
    def generate(cls) -> Self:
        return cls(f"{cls.prefix}_{uuid4().hex}")

    @classmethod
    def from_uuid(cls, value: UUID) -> Self:
        return cls(f"{cls.prefix}_{value.hex}")


class RunId(StableId):
    prefix = "run"


class CaseId(StableId):
    prefix = "case"


class ExperimentId(StableId):
    prefix = "experiment"


class MemberId(StableId):
    prefix = "member"


class ReportStepId(StableId):
    prefix = "report"


class MinistepId(StableId):
    prefix = "ministep"


class AttemptId(StableId):
    prefix = "attempt"


class CheckpointId(StableId):
    prefix = "checkpoint"


__all__ = [
    "StableId", "RunId", "CaseId", "ExperimentId", "MemberId",
    "ReportStepId", "MinistepId", "AttemptId", "CheckpointId",
]
