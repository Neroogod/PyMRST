"""Versioned contracts for workspace backup and recovery qualification."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ._common import nonempty


@dataclass(frozen=True, slots=True)
class BackupPlan:
    project_count: int
    file_count: int
    estimated_bytes: int
    database_schema_version: int
    includes: tuple[str, ...]

    def __post_init__(self):
        for name in ("project_count", "file_count", "estimated_bytes",
                     "database_schema_version"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)
        includes = tuple(nonempty(item, "backup include") for item in self.includes)
        if len(includes) != len(set(includes)):
            raise ValueError("backup includes must be unique")
        object.__setattr__(self, "includes", includes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_count": self.project_count, "file_count": self.file_count,
            "estimated_bytes": self.estimated_bytes,
            "database_schema_version": self.database_schema_version,
            "includes": list(self.includes),
        }


@dataclass(frozen=True, slots=True)
class BackupRecord:
    backup_id: str
    relative_path: str
    content_hash: str
    size_bytes: int
    project_count: int
    file_count: int
    status: str
    manifest: dict[str, Any]
    created_at: str
    verified_at: str
    pruned_at: str = ""

    def __post_init__(self):
        for name in ("backup_id", "relative_path", "status", "created_at"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        digest = str(self.content_hash).lower()
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("backup content_hash must be SHA-256")
        object.__setattr__(self, "content_hash", digest)
        for name in ("size_bytes", "project_count", "file_count"):
            value = int(getattr(self, name))
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
            object.__setattr__(self, name, value)
        if self.status not in {"created", "verified", "failed"}:
            raise ValueError("invalid backup status")
        object.__setattr__(self, "manifest", dict(self.manifest))
        object.__setattr__(self, "verified_at", str(self.verified_at).strip())
        object.__setattr__(self, "pruned_at", str(self.pruned_at).strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "backup_id": self.backup_id, "relative_path": self.relative_path,
            "content_hash": self.content_hash, "size_bytes": self.size_bytes,
            "project_count": self.project_count, "file_count": self.file_count,
            "status": self.status, "manifest": dict(self.manifest),
            "created_at": self.created_at, "verified_at": self.verified_at,
            "pruned_at": self.pruned_at,
        }


class BackupJobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class BackupJobRecord:
    job_id: str
    status: BackupJobStatus
    progress: float
    message: str
    backup_id: str
    error: dict[str, Any] | None
    cancellation_requested: bool
    created_at: str
    updated_at: str

    def __post_init__(self):
        for name in ("job_id", "message", "created_at", "updated_at"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        object.__setattr__(self, "status", BackupJobStatus(self.status))
        progress = float(self.progress)
        if not 0 <= progress <= 1:
            raise ValueError("backup job progress must lie in [0, 1]")
        object.__setattr__(self, "progress", progress)
        object.__setattr__(self, "backup_id", str(self.backup_id).strip())
        object.__setattr__(self, "error", None if self.error is None else dict(self.error))

    def to_dict(self):
        return {
            "job_id": self.job_id, "status": self.status.value,
            "progress": self.progress, "message": self.message,
            "backup_id": self.backup_id, "error": self.error,
            "cancellation_requested": self.cancellation_requested,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class BackupPolicy:
    enabled: bool = False
    interval_hours: float = 24.0
    retained_verified_backups: int = 10
    auto_verify: bool = True
    recovery_drill: bool = False

    def __post_init__(self):
        interval = float(self.interval_hours)
        retained = int(self.retained_verified_backups)
        if not 1 <= interval <= 8760:
            raise ValueError("backup interval must be between 1 hour and 1 year")
        if not 1 <= retained <= 1000:
            raise ValueError("backup retention count must be between 1 and 1000")
        object.__setattr__(self, "interval_hours", interval)
        object.__setattr__(self, "retained_verified_backups", retained)

    def to_dict(self):
        return {
            "enabled": bool(self.enabled), "interval_hours": self.interval_hours,
            "retained_verified_backups": self.retained_verified_backups,
            "auto_verify": bool(self.auto_verify),
            "recovery_drill": bool(self.recovery_drill),
        }

    @classmethod
    def from_dict(cls, value):
        if set(value) != {
            "enabled", "interval_hours", "retained_verified_backups",
            "auto_verify", "recovery_drill",
        }:
            raise ValueError("invalid BackupPolicy fields")
        return cls(**value)


@dataclass(frozen=True, slots=True)
class BackupQualificationRecord:
    qualification_id: str
    backup_id: str
    status: str
    checks: dict[str, Any]
    created_at: str

    def to_dict(self):
        return {
            "qualification_id": self.qualification_id,
            "backup_id": self.backup_id, "status": self.status,
            "checks": dict(self.checks), "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class RestoreReceipt:
    backup_id: str
    target_root: str
    archive_hash: str
    file_count: int
    restored_bytes: int
    database_integrity: str
    completed_at: str

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__slots__}


__all__ = [
    "BackupJobRecord", "BackupJobStatus", "BackupPlan", "BackupPolicy",
    "BackupQualificationRecord", "BackupRecord", "RestoreReceipt",
]
