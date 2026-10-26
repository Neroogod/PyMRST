"""Persistent schema for claim-to-evidence engineering reports."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping

from .._serialization import require_fields
from .evidence import EvidenceReference
from .findings import EngineeringFinding


ENGINEERING_REPORT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class ReportSection:
    section_id: str
    title: str
    narrative: str
    finding_ids: tuple[str, ...] = ()
    evidence: tuple[EvidenceReference, ...] = ()

    def __post_init__(self) -> None:
        for name in ("section_id", "title", "narrative"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        identifiers = tuple(str(item).strip() for item in self.finding_ids)
        if any(not item for item in identifiers) or len(set(identifiers)) != len(identifiers):
            raise ValueError("section finding ids must be unique and nonempty")
        object.__setattr__(self, "finding_ids", identifiers)
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return {
            "section_id": self.section_id, "title": self.title,
            "narrative": self.narrative,
            "finding_ids": list(self.finding_ids),
            "evidence": [item.to_dict() for item in self.evidence],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ReportSection":
        data = require_fields(
            value,
            required={"section_id", "title", "narrative", "finding_ids", "evidence"},
            name="ReportSection",
        )
        data["finding_ids"] = tuple(data["finding_ids"])
        data["evidence"] = tuple(EvidenceReference.from_dict(x) for x in data["evidence"])
        return cls(**data)


@dataclass(frozen=True, slots=True)
class EngineeringReport:
    report_id: str
    investigation_id: str
    title: str
    executive_summary: str
    findings: tuple[EngineeringFinding, ...]
    sections: tuple[ReportSection, ...]
    assumptions: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    qualified: bool = False
    schema_version: int = ENGINEERING_REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != ENGINEERING_REPORT_SCHEMA_VERSION:
            raise ValueError("unsupported EngineeringReport schema version")
        for name in ("report_id", "investigation_id", "title", "executive_summary"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        findings = tuple(self.findings)
        ids = [item.finding_id for item in findings]
        if len(set(ids)) != len(ids):
            raise ValueError("report finding ids must be unique")
        sections = tuple(self.sections)
        section_ids = [item.section_id for item in sections]
        if not sections or len(set(section_ids)) != len(section_ids):
            raise ValueError("report sections must be nonempty with unique ids")
        unknown = sorted({fid for section in sections for fid in section.finding_ids} - set(ids))
        if unknown:
            raise ValueError(f"report sections reference unknown findings: {unknown}")
        object.__setattr__(self, "findings", findings)
        object.__setattr__(self, "sections", sections)
        for name in ("assumptions", "limitations"):
            values = tuple(str(item).strip() for item in getattr(self, name))
            if any(not item for item in values):
                raise ValueError(f"{name} cannot contain empty entries")
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version, "report_id": self.report_id,
            "investigation_id": self.investigation_id, "title": self.title,
            "executive_summary": self.executive_summary,
            "findings": [item.to_dict() for item in self.findings],
            "sections": [item.to_dict() for item in self.sections],
            "assumptions": list(self.assumptions),
            "limitations": list(self.limitations), "qualified": self.qualified,
        }

    @property
    def content_hash(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"),
            ensure_ascii=False, allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(
            self.to_dict(), indent=indent, sort_keys=True,
            ensure_ascii=False, allow_nan=False,
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "EngineeringReport":
        fields = {
            "schema_version", "report_id", "investigation_id", "title",
            "executive_summary", "findings", "sections", "assumptions",
            "limitations", "qualified",
        }
        data = require_fields(value, required=fields, name="EngineeringReport")
        data["findings"] = tuple(EngineeringFinding.from_dict(x) for x in data["findings"])
        data["sections"] = tuple(ReportSection.from_dict(x) for x in data["sections"])
        data["assumptions"] = tuple(data["assumptions"])
        data["limitations"] = tuple(data["limitations"])
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> "EngineeringReport":
        parsed = json.loads(value)
        if not isinstance(parsed, dict):
            raise ValueError("EngineeringReport JSON root must be an object")
        return cls.from_dict(parsed)


__all__ = [
    "ENGINEERING_REPORT_SCHEMA_VERSION", "EngineeringReport", "ReportSection",
]
