"""Persistent investigation memory tied to case and run lineage."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import json
from pathlib import Path
from typing import Any, Mapping

from ._serialization import frozen_json_mapping, json_value, require_fields
from .base import ToolCall, ToolResult
from .context import InvestigationContext
from .schemas.findings import EngineeringFinding


INVESTIGATION_RECORD_SCHEMA_VERSION = 1


class InvestigationStatus(StrEnum):
    OPEN = "open"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    UNQUALIFIED = "unqualified"
    CANCELLED = "cancelled"


class EntryKind(StrEnum):
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    FINDING = "finding"
    HUMAN_DECISION = "human_decision"
    POLICY_DECISION = "policy_decision"
    REPORT = "report"
    EXECUTION_GRAPH = "execution_graph"
    CONFLICT = "conflict"
    UNCERTAINTY = "uncertainty"
    NOTE = "note"


@dataclass(frozen=True, slots=True)
class InvestigationEntry:
    sequence: int
    kind: EntryKind
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        sequence = int(self.sequence)
        if sequence < 0:
            raise ValueError("investigation entry sequence must be nonnegative")
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(self, "kind", EntryKind(self.kind))
        object.__setattr__(self, "payload", frozen_json_mapping(self.payload, "$.payload"))

    def to_dict(self) -> dict[str, Any]:
        return json_value({
            "sequence": self.sequence, "kind": self.kind.value,
            "payload": self.payload,
        })

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InvestigationEntry":
        return cls(**require_fields(
            value, required={"sequence", "kind", "payload"},
            name="InvestigationEntry",
        ))


@dataclass(frozen=True, slots=True)
class InvestigationRecord:
    context: InvestigationContext
    status: InvestigationStatus = InvestigationStatus.OPEN
    entries: tuple[InvestigationEntry, ...] = ()
    schema_version: int = INVESTIGATION_RECORD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != INVESTIGATION_RECORD_SCHEMA_VERSION:
            raise ValueError("unsupported InvestigationRecord schema version")
        object.__setattr__(self, "status", InvestigationStatus(self.status))
        entries = tuple(self.entries)
        expected = tuple(range(len(entries)))
        if tuple(entry.sequence for entry in entries) != expected:
            raise ValueError("investigation entry sequences must be contiguous from zero")
        object.__setattr__(self, "entries", entries)

    def append(self, kind: EntryKind, payload: Mapping[str, Any]) -> "InvestigationRecord":
        entry = InvestigationEntry(len(self.entries), kind, payload)
        return InvestigationRecord(self.context, self.status, self.entries + (entry,))

    def with_status(self, status: InvestigationStatus) -> "InvestigationRecord":
        return InvestigationRecord(self.context, status, self.entries)

    def record_call(self, call: ToolCall) -> "InvestigationRecord":
        return self.append(EntryKind.TOOL_CALL, call.to_dict())

    def record_result(self, result: ToolResult) -> "InvestigationRecord":
        return self.append(EntryKind.TOOL_RESULT, result.to_dict())

    def record_finding(self, finding: EngineeringFinding) -> "InvestigationRecord":
        return self.append(EntryKind.FINDING, finding.to_dict())

    def record_policy_decision(self, finding_id: str, decision) -> "InvestigationRecord":
        return self.append(EntryKind.POLICY_DECISION, {
            "finding_id": str(finding_id), "decision": decision.to_dict(),
        })

    def record_report(self, report) -> "InvestigationRecord":
        return self.append(EntryKind.REPORT, report.to_dict())

    def record_human_decision(self, decision) -> "InvestigationRecord":
        return self.append(EntryKind.HUMAN_DECISION, decision.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "context": self.context.to_dict(), "status": self.status.value,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    def to_json(self, *, indent: int | None = None) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True, allow_nan=False)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "InvestigationRecord":
        data = require_fields(
            value, required={"schema_version", "context", "status", "entries"},
            name="InvestigationRecord",
        )
        data["context"] = InvestigationContext.from_dict(data["context"])
        data["entries"] = tuple(InvestigationEntry.from_dict(x) for x in data["entries"])
        return cls(**data)

    @classmethod
    def from_json(cls, value: str) -> "InvestigationRecord":
        data = json.loads(value)
        if not isinstance(data, dict):
            raise ValueError("InvestigationRecord JSON root must be an object")
        return cls.from_dict(data)


def write_investigation(path: str | Path, record: InvestigationRecord) -> Path:
    """Atomically persist a complete record without touching simulator storage."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(record.to_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


__all__ = [
    "EntryKind", "INVESTIGATION_RECORD_SCHEMA_VERSION", "InvestigationEntry",
    "InvestigationRecord", "InvestigationStatus", "write_investigation",
]
