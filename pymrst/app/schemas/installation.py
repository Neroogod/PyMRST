"""First-launch and installed-runtime qualification contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ._common import json_value, nonempty


class QualificationCheckStatus(StrEnum):
    PASSED = "passed"
    WARNING = "warning"
    FAILED = "failed"


@dataclass(frozen=True, slots=True)
class EnvironmentCheck:
    check_id: str
    category: str
    title: str
    status: QualificationCheckStatus
    required: bool
    message: str
    details: dict[str, Any]
    remediation: str = ""

    def __post_init__(self):
        for name in ("check_id", "category", "title", "message"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        object.__setattr__(self, "status", QualificationCheckStatus(self.status))
        object.__setattr__(self, "details", json_value(dict(self.details)))
        object.__setattr__(self, "remediation", str(self.remediation).strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id, "category": self.category,
            "title": self.title, "status": self.status.value,
            "required": self.required, "message": self.message,
            "details": dict(self.details), "remediation": self.remediation,
        }


@dataclass(frozen=True, slots=True)
class EnvironmentQualification:
    qualification_id: str
    status: str
    content_hash: str
    platform: dict[str, Any]
    checks: tuple[EnvironmentCheck, ...]
    created_at: str

    def __post_init__(self):
        object.__setattr__(self, "qualification_id", nonempty(
            self.qualification_id, "qualification_id",
        ))
        if self.status not in {"ready", "degraded", "blocked"}:
            raise ValueError("invalid environment qualification status")
        digest = str(self.content_hash).lower()
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("qualification content_hash must be SHA-256")
        object.__setattr__(self, "content_hash", digest)
        object.__setattr__(self, "platform", json_value(dict(self.platform)))
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "created_at", nonempty(self.created_at, "created_at"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "qualification_id": self.qualification_id, "status": self.status,
            "content_hash": self.content_hash, "platform": dict(self.platform),
            "checks": [item.to_dict() for item in self.checks],
            "summary": {
                state.value: sum(item.status is state for item in self.checks)
                for state in QualificationCheckStatus
            },
            "created_at": self.created_at,
        }


__all__ = [
    "EnvironmentCheck", "EnvironmentQualification", "QualificationCheckStatus",
]
