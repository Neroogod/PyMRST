"""Bounded, evidence-aware context passed to engineering agents."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from pymrst.research_studio.runtime.identifiers import CaseId, RunId

from ._serialization import frozen_json_mapping, json_value, require_fields
from .permissions import PermissionGrant
from .schemas.evidence import EvidenceReference


AGENT_CONTEXT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class Selection:
    cells: tuple[int, ...] = ()
    faces: tuple[int, ...] = ()
    wells: tuple[str, ...] = ()
    regions: tuple[str, ...] = ()
    report_steps: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for name in ("cells", "faces", "report_steps"):
            values = tuple(int(item) for item in getattr(self, name))
            if any(item < 0 for item in values) or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique nonnegative indices")
            object.__setattr__(self, name, values)
        for name in ("wells", "regions"):
            values = tuple(str(item).strip() for item in getattr(self, name))
            if any(not item for item in values) or len(set(values)) != len(values):
                raise ValueError(f"{name} must contain unique nonempty identifiers")
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cells": list(self.cells), "faces": list(self.faces),
            "wells": list(self.wells), "regions": list(self.regions),
            "report_steps": list(self.report_steps),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Selection":
        return cls(**require_fields(
            value,
            required={"cells", "faces", "wells", "regions", "report_steps"},
            name="Selection",
        ))


@dataclass(frozen=True, slots=True)
class InvestigationContext:
    investigation_id: str
    question: str
    case_id: CaseId
    case_content_hash: str
    permission: PermissionGrant
    run_ids: tuple[RunId, ...] = ()
    selection: Selection = Selection()
    evidence: tuple[EvidenceReference, ...] = ()
    attributes: Mapping[str, Any] = field(default_factory=dict)
    max_evidence_items: int = 1_000
    schema_version: int = AGENT_CONTEXT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != AGENT_CONTEXT_SCHEMA_VERSION:
            raise ValueError("unsupported InvestigationContext schema version")
        for name in ("investigation_id", "question"):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} cannot be empty")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "case_id", CaseId(self.case_id))
        digest = str(self.case_content_hash).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("case_content_hash must be a lowercase SHA-256 digest")
        object.__setattr__(self, "case_content_hash", digest)
        run_ids = tuple(RunId(item) for item in self.run_ids)
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("run_ids must be unique")
        object.__setattr__(self, "run_ids", run_ids)
        evidence = tuple(
            item if isinstance(item, EvidenceReference)
            else EvidenceReference.from_dict(item) for item in self.evidence
        )
        limit = int(self.max_evidence_items)
        if limit <= 0 or len(evidence) > limit:
            raise ValueError("evidence exceeds the positive context evidence limit")
        object.__setattr__(self, "max_evidence_items", limit)
        object.__setattr__(self, "evidence", evidence)
        object.__setattr__(
            self, "attributes", frozen_json_mapping(self.attributes, "$.attributes")
        )

    def to_dict(self) -> dict[str, Any]:
        return json_value({
            "schema_version": self.schema_version,
            "investigation_id": self.investigation_id,
            "question": self.question,
            "case_id": str(self.case_id),
            "case_content_hash": self.case_content_hash,
            "permission": self.permission.to_dict(),
            "run_ids": [str(item) for item in self.run_ids],
            "selection": self.selection.to_dict(),
            "evidence": [item.to_dict() for item in self.evidence],
            "attributes": self.attributes,
            "max_evidence_items": self.max_evidence_items,
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InvestigationContext":
        data = require_fields(
            value,
            required={
                "schema_version", "investigation_id", "question", "case_id",
                "case_content_hash", "permission", "run_ids", "selection",
                "evidence", "attributes", "max_evidence_items",
            },
            name="InvestigationContext",
        )
        data["permission"] = PermissionGrant.from_dict(data["permission"])
        data["selection"] = Selection.from_dict(data["selection"])
        data["evidence"] = tuple(
            EvidenceReference.from_dict(item) for item in data["evidence"]
        )
        return cls(**data)


__all__ = [
    "AGENT_CONTEXT_SCHEMA_VERSION", "InvestigationContext", "Selection",
]
