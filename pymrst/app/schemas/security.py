"""Authentication, authorization, and security-audit contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ._common import json_value, nonempty


class DeploymentMode(StrEnum):
    LOCAL = "local"
    SHARED = "shared"


class SystemRole(StrEnum):
    ADMIN = "admin"
    ENGINEER = "engineer"
    VIEWER = "viewer"
    AUDITOR = "auditor"


class ProjectRole(StrEnum):
    OWNER = "owner"
    EDITOR = "editor"
    RUNNER = "runner"
    ANALYST = "analyst"
    VIEWER = "viewer"


class Capability(StrEnum):
    SYSTEM_ADMIN = "system_admin"
    VIEW = "view"
    EDIT = "edit"
    EXECUTE = "execute"
    PUBLISH = "publish"
    AUDIT = "audit"


ALL_CAPABILITIES = tuple(Capability)


@dataclass(frozen=True, slots=True)
class SecurityActor:
    user_id: str
    display_name: str
    system_role: SystemRole
    token_id: str
    token_scopes: tuple[Capability, ...]
    local_session: bool = False

    def __post_init__(self):
        for name in ("user_id", "display_name"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        object.__setattr__(self, "system_role", SystemRole(self.system_role))
        object.__setattr__(self, "token_id", str(self.token_id).strip())
        scopes = tuple(Capability(item) for item in self.token_scopes)
        if len(scopes) != len(set(scopes)):
            raise ValueError("actor token scopes must be unique")
        object.__setattr__(self, "token_scopes", scopes)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id, "display_name": self.display_name,
            "system_role": self.system_role.value, "token_id": self.token_id,
            "token_scopes": [item.value for item in self.token_scopes],
            "local_session": self.local_session,
        }


@dataclass(frozen=True, slots=True)
class UserRecord:
    user_id: str
    display_name: str
    email: str
    system_role: SystemRole
    active: bool
    created_at: str
    updated_at: str

    def __post_init__(self):
        for name in ("user_id", "display_name", "email", "created_at", "updated_at"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        email = self.email.casefold()
        if "@" not in email or email.startswith("@") or email.endswith("@"):
            raise ValueError("email must be a valid address")
        object.__setattr__(self, "email", email)
        object.__setattr__(self, "system_role", SystemRole(self.system_role))

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id, "display_name": self.display_name,
            "email": self.email, "system_role": self.system_role.value,
            "active": self.active, "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class ProjectMembershipRecord:
    project_id: str
    user_id: str
    project_role: ProjectRole
    created_at: str
    updated_at: str

    def __post_init__(self):
        for name in ("project_id", "user_id", "created_at", "updated_at"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        object.__setattr__(self, "project_role", ProjectRole(self.project_role))

    def to_dict(self) -> dict[str, Any]:
        return {
            "project_id": self.project_id, "user_id": self.user_id,
            "project_role": self.project_role.value,
            "created_at": self.created_at, "updated_at": self.updated_at,
        }


@dataclass(frozen=True, slots=True)
class AccessTokenRecord:
    token_id: str
    user_id: str
    label: str
    token_prefix: str
    scopes: tuple[Capability, ...]
    expires_at: str
    revoked_at: str
    last_used_at: str
    created_at: str

    def __post_init__(self):
        for name in ("token_id", "user_id", "label", "token_prefix", "created_at"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        scopes = tuple(Capability(item) for item in self.scopes)
        if not scopes or len(scopes) != len(set(scopes)):
            raise ValueError("access-token scopes must be non-empty and unique")
        object.__setattr__(self, "scopes", scopes)
        for name in ("expires_at", "revoked_at", "last_used_at"):
            object.__setattr__(self, name, str(getattr(self, name)).strip())

    @property
    def revoked(self) -> bool:
        return bool(self.revoked_at)

    def to_dict(self) -> dict[str, Any]:
        return {
            "token_id": self.token_id, "user_id": self.user_id,
            "label": self.label, "token_prefix": self.token_prefix,
            "scopes": [item.value for item in self.scopes],
            "expires_at": self.expires_at, "revoked_at": self.revoked_at,
            "last_used_at": self.last_used_at, "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class IssuedAccessToken:
    record: AccessTokenRecord
    token: str

    def __post_init__(self):
        object.__setattr__(self, "token", nonempty(self.token, "token"))

    def to_dict(self) -> dict[str, Any]:
        return {**self.record.to_dict(), "token": self.token, "shown_once": True}


@dataclass(frozen=True, slots=True)
class SecurityAuditEvent:
    sequence: int
    event_id: str
    actor_id: str
    action: str
    resource_type: str
    resource_id: str
    project_id: str
    outcome: str
    details: dict[str, Any]
    previous_hash: str
    event_hash: str
    created_at: str

    def __post_init__(self):
        for name in ("event_id", "actor_id", "action", "resource_type", "outcome", "created_at"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        if int(self.sequence) < 1:
            raise ValueError("audit sequence must be positive")
        object.__setattr__(self, "sequence", int(self.sequence))
        if self.outcome not in {"allowed", "denied", "succeeded", "failed"}:
            raise ValueError("invalid audit outcome")
        for name in ("previous_hash", "event_hash"):
            digest = str(getattr(self, name)).lower()
            if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
                raise ValueError(f"{name} must be SHA-256")
            object.__setattr__(self, name, digest)
        object.__setattr__(self, "resource_id", str(self.resource_id).strip())
        object.__setattr__(self, "project_id", str(self.project_id).strip())
        object.__setattr__(self, "details", json_value(dict(self.details)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence": self.sequence, "event_id": self.event_id,
            "actor_id": self.actor_id, "action": self.action,
            "resource_type": self.resource_type, "resource_id": self.resource_id,
            "project_id": self.project_id, "outcome": self.outcome,
            "details": dict(self.details), "previous_hash": self.previous_hash,
            "event_hash": self.event_hash, "created_at": self.created_at,
        }


__all__ = [
    "ALL_CAPABILITIES", "AccessTokenRecord", "Capability", "DeploymentMode",
    "IssuedAccessToken", "ProjectMembershipRecord", "ProjectRole",
    "SecurityActor", "SecurityAuditEvent", "SystemRole", "UserRecord",
]
