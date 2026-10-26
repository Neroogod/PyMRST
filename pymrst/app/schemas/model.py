"""Immutable model-inspection, validation, and input-patch contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping

from ._common import json_value, nonempty


class IssueSeverity(StrEnum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


@dataclass(frozen=True, slots=True)
class SourceLocation:
    asset_id: str
    path: str
    line: int
    column: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "asset_id", nonempty(self.asset_id, "asset_id"))
        object.__setattr__(self, "path", nonempty(self.path, "source path"))
        if int(self.line) < 1 or int(self.column) < 1:
            raise ValueError("source line and column are one-based positive integers")
        object.__setattr__(self, "line", int(self.line))
        object.__setattr__(self, "column", int(self.column))

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    severity: IssueSeverity
    category: str
    message: str
    location: SourceLocation | None = None
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in ("code", "category", "message"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        object.__setattr__(self, "severity", IssueSeverity(self.severity))
        object.__setattr__(self, "details", json_value(dict(self.details or {})))

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code, "severity": self.severity.value,
            "category": self.category, "message": self.message,
            "location": None if self.location is None else self.location.to_dict(),
            "details": dict(self.details or {}),
        }


@dataclass(frozen=True, slots=True)
class DeckKeywordOccurrence:
    occurrence_id: str
    keyword: str
    section: str
    location: SourceLocation
    end_line: int
    record_count: int
    value_count: int
    defaulted_value_count: int

    def __post_init__(self) -> None:
        for name in ("occurrence_id", "keyword"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        object.__setattr__(self, "keyword", self.keyword.upper())
        object.__setattr__(self, "section", str(self.section).upper())
        for name in ("end_line", "record_count", "value_count", "defaulted_value_count"):
            value = int(getattr(self, name))
            if value < 0 or (name == "end_line" and value < self.location.line):
                raise ValueError(f"invalid {name}")
            object.__setattr__(self, name, value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurrence_id": self.occurrence_id, "keyword": self.keyword,
            "section": self.section, "location": self.location.to_dict(),
            "end_line": self.end_line, "record_count": self.record_count,
            "value_count": self.value_count,
            "defaulted_value_count": self.defaulted_value_count,
        }


@dataclass(frozen=True, slots=True)
class ModelInspection:
    model_id: str
    project_id: str
    asset_id: str
    name: str
    kind: str
    unit_system: str
    status: str
    source_hash: str
    inspection_hash: str
    sections: tuple[str, ...]
    files: tuple[Mapping[str, Any], ...]
    occurrences: tuple[DeckKeywordOccurrence, ...]
    issues: tuple[ValidationIssue, ...]
    created_at: str
    updated_at: str

    def to_dict(self, *, include_occurrences: bool = True) -> dict[str, Any]:
        value = {
            "model_id": self.model_id, "project_id": self.project_id,
            "asset_id": self.asset_id, "name": self.name, "kind": self.kind,
            "unit_system": self.unit_system, "status": self.status,
            "source_hash": self.source_hash, "inspection_hash": self.inspection_hash,
            "sections": list(self.sections), "files": [dict(item) for item in self.files],
            "issues": [item.to_dict() for item in self.issues],
            "created_at": self.created_at, "updated_at": self.updated_at,
        }
        if include_occurrences:
            value["occurrences"] = [item.to_dict() for item in self.occurrences]
        return value


class PatchOperationKind(StrEnum):
    REPLACE_LINES = "replace_lines"
    INSERT_BEFORE = "insert_before"
    INSERT_AFTER = "insert_after"
    DELETE_LINES = "delete_lines"


@dataclass(frozen=True, slots=True)
class PatchOperation:
    kind: PatchOperationKind
    start_line: int
    end_line: int
    text: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", PatchOperationKind(self.kind))
        start, end = int(self.start_line), int(self.end_line)
        if start < 1 or end < start:
            raise ValueError("patch line range must be one-based and ordered")
        if self.kind is PatchOperationKind.DELETE_LINES and self.text:
            raise ValueError("delete operation cannot contain replacement text")
        object.__setattr__(self, "start_line", start)
        object.__setattr__(self, "end_line", end)
        object.__setattr__(self, "text", str(self.text))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value, "start_line": self.start_line,
            "end_line": self.end_line, "text": self.text,
        }


@dataclass(frozen=True, slots=True)
class InputPatch:
    patch_id: str
    project_id: str
    model_id: str
    title: str
    description: str
    base_hash: str
    patch_hash: str
    operations: tuple[PatchOperation, ...]
    status: str
    variant_relative_path: str
    variant_hash: str
    created_at: str
    materialized_at: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "patch_id": self.patch_id, "project_id": self.project_id,
            "model_id": self.model_id, "title": self.title,
            "description": self.description, "base_hash": self.base_hash,
            "patch_hash": self.patch_hash,
            "operations": [item.to_dict() for item in self.operations],
            "status": self.status, "variant_relative_path": self.variant_relative_path,
            "variant_hash": self.variant_hash, "created_at": self.created_at,
            "materialized_at": self.materialized_at,
        }


__all__ = [
    "DeckKeywordOccurrence", "InputPatch", "IssueSeverity", "ModelInspection",
    "PatchOperation", "PatchOperationKind", "SourceLocation", "ValidationIssue",
]
