"""Opt-in shared-deployment security with least privilege and auditable decisions."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import hmac
import json
import secrets
from typing import Any, Iterable
from uuid import uuid4

from ..database.models import AppDatabase
from ..schemas._common import json_value, nonempty, utc_now
from ..schemas.security import (
    ALL_CAPABILITIES, AccessTokenRecord, Capability, DeploymentMode,
    IssuedAccessToken, ProjectMembershipRecord, ProjectRole, SecurityActor,
    SecurityAuditEvent, SystemRole, UserRecord,
)


_MODE_SETTING = "security.deployment_mode"
_GENESIS_HASH = "0" * 64
_SYSTEM_CAPABILITIES = {
    SystemRole.ADMIN: frozenset(ALL_CAPABILITIES),
    SystemRole.AUDITOR: frozenset({Capability.VIEW, Capability.AUDIT}),
    SystemRole.ENGINEER: frozenset(),
    SystemRole.VIEWER: frozenset(),
}
_PROJECT_CAPABILITIES = {
    ProjectRole.OWNER: frozenset({
        Capability.VIEW, Capability.EDIT, Capability.EXECUTE,
        Capability.PUBLISH, Capability.AUDIT,
    }),
    ProjectRole.EDITOR: frozenset({
        Capability.VIEW, Capability.EDIT, Capability.EXECUTE, Capability.PUBLISH,
    }),
    ProjectRole.RUNNER: frozenset({Capability.VIEW, Capability.EXECUTE}),
    ProjectRole.ANALYST: frozenset({Capability.VIEW, Capability.PUBLISH}),
    ProjectRole.VIEWER: frozenset({Capability.VIEW}),
}


class AuthenticationError(PermissionError):
    """The supplied credential is missing, invalid, expired, or revoked."""


class AuthorizationError(PermissionError):
    """The authenticated principal lacks a required capability."""


def _json(value: Any) -> str:
    return json.dumps(json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def _parse_time(value: str) -> datetime | None:
    value = str(value).strip()
    if not value:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("security timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def _user(row) -> UserRecord:
    return UserRecord(
        row["user_id"], row["display_name"], row["email"], row["system_role"],
        bool(row["active"]), row["created_at"], row["updated_at"],
    )


def _membership(row) -> ProjectMembershipRecord:
    return ProjectMembershipRecord(
        row["project_id"], row["user_id"], row["project_role"],
        row["created_at"], row["updated_at"],
    )


def _token(row) -> AccessTokenRecord:
    return AccessTokenRecord(
        row["token_id"], row["user_id"], row["label"], row["token_prefix"],
        tuple(json.loads(row["scopes_json"])), row["expires_at"], row["revoked_at"],
        row["last_used_at"], row["created_at"],
    )


def _audit(row) -> SecurityAuditEvent:
    return SecurityAuditEvent(
        row["sequence"], row["event_id"], row["actor_id"], row["action"],
        row["resource_type"], row["resource_id"], row["project_id"], row["outcome"],
        json.loads(row["details_json"]), row["previous_hash"], row["event_hash"],
        row["created_at"],
    )


def _safe_details(value: Any, path: str = "details") -> Any:
    """Reject credential-like fields rather than attempting fragile redaction."""
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if any(word in normalized for word in ("password", "secret", "token", "credential")):
                raise ValueError(f"security audit details cannot contain sensitive field {path}.{key}")
            result[str(key)] = _safe_details(child, f"{path}.{key}")
        return result
    if isinstance(value, (tuple, list)):
        return [_safe_details(child, f"{path}[{index}]") for index, child in enumerate(value)]
    return json_value(value, path)


class SecurityService:
    """Security authority for API and application-service boundaries.

    Local mode deliberately preserves the single-user desktop experience. Shared
    mode rejects unauthenticated access and applies token-scoped RBAC.
    """

    def __init__(self, database: AppDatabase) -> None:
        self.database = database

    @property
    def mode(self) -> DeploymentMode:
        row = self.database.fetch_one(
            "SELECT value_json FROM app_settings WHERE setting_key=?", (_MODE_SETTING,),
        )
        if row is None:
            return DeploymentMode.LOCAL
        return DeploymentMode(json.loads(row["value_json"]))

    @staticmethod
    def local_actor() -> SecurityActor:
        return SecurityActor(
            "local_operator", "Local desktop operator", SystemRole.ADMIN, "",
            ALL_CAPABILITIES, True,
        )

    def actor_or_local(self, actor: SecurityActor | None) -> SecurityActor:
        if actor is not None:
            if actor.local_session and self.mode is DeploymentMode.SHARED:
                raise AuthenticationError("local desktop identity is disabled in shared mode")
            return actor
        if self.mode is DeploymentMode.LOCAL:
            return self.local_actor()
        raise AuthenticationError("shared deployment requires a bearer token")

    def create_user(
        self, display_name: str, email: str, system_role: SystemRole | str,
        *, actor: SecurityActor | None = None,
    ) -> UserRecord:
        principal = self.actor_or_local(actor)
        self.require(principal, Capability.SYSTEM_ADMIN, action="security.user.create")
        now = utc_now()
        record = UserRecord(
            f"user_{uuid4().hex}", display_name, email, SystemRole(system_role),
            True, now, now,
        )
        try:
            with self.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO app_users VALUES(?,?,?,?,?,?,?)",
                    (record.user_id, record.display_name, record.email,
                     record.system_role.value, 1, now, now),
                )
            self.audit(principal, "security.user.create", "user", record.user_id,
                       outcome="succeeded", details={"system_role": record.system_role.value})
        except Exception as error:
            self.audit(principal, "security.user.create", "user", "", outcome="failed",
                       details={"error_type": type(error).__name__})
            raise
        return record

    def get_user(self, user_id: str) -> UserRecord:
        row = self.database.fetch_one("SELECT * FROM app_users WHERE user_id=?", (user_id,))
        if row is None:
            raise KeyError(f"unknown application user: {user_id}")
        return _user(row)

    def users(self) -> tuple[UserRecord, ...]:
        return tuple(_user(row) for row in self.database.fetch_all(
            "SELECT * FROM app_users ORDER BY display_name COLLATE NOCASE, user_id"
        ))

    def set_user_active(
        self, user_id: str, active: bool, *, actor: SecurityActor | None = None,
    ) -> UserRecord:
        principal = self.actor_or_local(actor)
        self.require(principal, Capability.SYSTEM_ADMIN, action="security.user.active")
        current = self.get_user(user_id)
        if not active and current.system_role is SystemRole.ADMIN and self.mode is DeploymentMode.SHARED:
            if self._viable_admin_count(excluding_user=user_id) == 0:
                raise ValueError("cannot disable the last viable shared-deployment administrator")
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE app_users SET active=?,updated_at=? WHERE user_id=?",
                (int(bool(active)), now, user_id),
            )
        self.audit(principal, "security.user.active", "user", user_id,
                   outcome="succeeded", details={"active": bool(active)})
        return self.get_user(user_id)

    def set_membership(
        self, project_id: str, user_id: str, project_role: ProjectRole | str,
        *, actor: SecurityActor | None = None,
    ) -> ProjectMembershipRecord:
        principal = self.actor_or_local(actor)
        self.require(principal, Capability.EDIT, project_id=project_id,
                     action="security.membership.set")
        if self.database.fetch_one("SELECT 1 FROM app_projects WHERE project_id=?", (project_id,)) is None:
            raise KeyError(f"unknown project: {project_id}")
        self.get_user(user_id)
        role, now = ProjectRole(project_role), utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO app_project_memberships VALUES(?,?,?,?,?) "
                "ON CONFLICT(project_id,user_id) DO UPDATE SET "
                "project_role=excluded.project_role,updated_at=excluded.updated_at",
                (project_id, user_id, role.value, now, now),
            )
        self.audit(principal, "security.membership.set", "membership", user_id,
                   project_id=project_id, outcome="succeeded",
                   details={"project_role": role.value})
        row = self.database.fetch_one(
            "SELECT * FROM app_project_memberships WHERE project_id=? AND user_id=?",
            (project_id, user_id),
        )
        return _membership(row)

    def memberships(self, user_id: str | None = None) -> tuple[ProjectMembershipRecord, ...]:
        if user_id is None:
            rows = self.database.fetch_all(
                "SELECT * FROM app_project_memberships ORDER BY project_id,user_id"
            )
        else:
            self.get_user(user_id)
            rows = self.database.fetch_all(
                "SELECT * FROM app_project_memberships WHERE user_id=? ORDER BY project_id",
                (user_id,),
            )
        return tuple(_membership(row) for row in rows)

    def visible_project_ids(self, actor: SecurityActor) -> tuple[str, ...]:
        if actor.local_session or actor.system_role in {SystemRole.ADMIN, SystemRole.AUDITOR}:
            return tuple(row["project_id"] for row in self.database.fetch_all(
                "SELECT project_id FROM app_projects ORDER BY project_id"
            ))
        return tuple(row["project_id"] for row in self.database.fetch_all(
            "SELECT project_id FROM app_project_memberships WHERE user_id=? ORDER BY project_id",
            (actor.user_id,),
        ))

    def project_for_resource(self, resource_kind: str, resource_id: str) -> str:
        """Resolve an API resource to its authorization owner without loading payloads."""
        mapping = {
            "projects": ("app_projects", "project_id", "project_id"),
            "models": ("app_models", "model_id", "project_id"),
            "patches": ("app_input_patches", "patch_id", "project_id"),
            "grids": ("app_grid_snapshots", "snapshot_id", "project_id"),
            "model-drafts": ("app_model_drafts", "draft_id", "project_id"),
            "model-revisions": ("app_model_revisions", "revision_id", "project_id"),
            "experiments": ("app_experiments", "experiment_id", "project_id"),
            "experiment-executions": (
                "app_experiment_executions", "execution_id", "project_id",
            ),
            "experiment-branches": ("app_experiment_branches", "branch_id", "project_id"),
            "investigations": ("app_investigations", "investigation_id", "project_id"),
            "investigation-executions": (
                "app_investigation_executions", "execution_id", "project_id",
            ),
            "findings": ("app_investigation_findings", "finding_id", "project_id"),
            "variant-proposals": ("app_variant_proposals", "proposal_id", "project_id"),
            "investigation-narratives": (
                "app_investigation_narratives", "narrative_id", "project_id",
            ),
            "run-configurations": (
                "app_run_configurations", "configuration_id", "project_id",
            ),
            "runs": ("app_runs", "run_id", "project_id"),
            "tasks": ("app_tasks", "task_id", "project_id"),
            "run-comparisons": (
                "app_result_comparisons", "comparison_id", "project_id",
            ),
        }
        if resource_kind == "result-states":
            row = self.database.fetch_one(
                "SELECT r.project_id FROM app_result_states s JOIN app_runs r "
                "ON r.run_id=s.run_id WHERE s.state_id=?", (resource_id,),
            )
            return "" if row is None else str(row["project_id"])
        target = mapping.get(str(resource_kind))
        if target is None:
            return ""
        table, identifier, project_column = target
        row = self.database.fetch_one(
            f"SELECT {project_column} AS project_id FROM {table} WHERE {identifier}=?",
            (resource_id,),
        )
        return "" if row is None else str(row["project_id"])

    def issue_token(
        self, user_id: str, label: str, scopes: Iterable[Capability | str],
        *, expires_at: str = "", actor: SecurityActor | None = None,
    ) -> IssuedAccessToken:
        principal = self.actor_or_local(actor)
        if principal.user_id != user_id:
            self.require(principal, Capability.SYSTEM_ADMIN, action="security.token.issue")
        user = self.get_user(user_id)
        if not user.active:
            raise ValueError("cannot issue a token for a disabled user")
        scope_values = tuple(Capability(item) for item in scopes)
        if not scope_values or len(scope_values) != len(set(scope_values)):
            raise ValueError("token scopes must be non-empty and unique")
        if Capability.SYSTEM_ADMIN in scope_values and user.system_role is not SystemRole.ADMIN:
            raise ValueError("system_admin scope is reserved for administrators")
        expiry = str(expires_at).strip()
        parsed_expiry = _parse_time(expiry)
        if parsed_expiry is not None and parsed_expiry <= datetime.now(timezone.utc):
            raise ValueError("token expiry must be in the future")
        token_id = f"pat_{uuid4().hex}"
        secret = secrets.token_urlsafe(32)
        plaintext = f"pymrst_{token_id}.{secret}"
        prefix = plaintext[:24]
        digest, now = hashlib.sha256(plaintext.encode()).hexdigest(), utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO app_access_tokens VALUES(?,?,?,?,?,?,?,?,?,?)",
                (token_id, user_id, nonempty(label, "token label"), prefix, digest,
                 _json([item.value for item in scope_values]), expiry, "", "", now),
            )
        self.audit(principal, "security.token.issue", "access_token", token_id,
                   outcome="succeeded", details={"user_id": user_id,
                   "scopes": [item.value for item in scope_values], "expires_at": expiry})
        return IssuedAccessToken(self.get_token(token_id), plaintext)

    def get_token(self, token_id: str) -> AccessTokenRecord:
        row = self.database.fetch_one(
            "SELECT * FROM app_access_tokens WHERE token_id=?", (token_id,),
        )
        if row is None:
            raise KeyError(f"unknown access token: {token_id}")
        return _token(row)

    def tokens(self, user_id: str | None = None) -> tuple[AccessTokenRecord, ...]:
        query, parameters = "SELECT * FROM app_access_tokens", ()
        if user_id is not None:
            self.get_user(user_id)
            query, parameters = query + " WHERE user_id=?", (user_id,)
        return tuple(_token(row) for row in self.database.fetch_all(
            query + " ORDER BY created_at", parameters,
        ))

    def authenticate(self, plaintext: str) -> SecurityActor:
        plaintext = str(plaintext).strip()
        if not plaintext.startswith("pymrst_pat_") or "." not in plaintext:
            raise AuthenticationError("invalid access-token format")
        token_id = plaintext[len("pymrst_"):].split(".", 1)[0]
        row = self.database.fetch_one(
            "SELECT t.*,u.display_name,u.system_role,u.active FROM app_access_tokens t "
            "JOIN app_users u ON u.user_id=t.user_id WHERE t.token_id=?", (token_id,),
        )
        supplied_hash = hashlib.sha256(plaintext.encode()).hexdigest()
        if row is None or not hmac.compare_digest(str(row["token_hash"]), supplied_hash):
            raise AuthenticationError("invalid access token")
        if not bool(row["active"]):
            raise AuthenticationError("access-token owner is disabled")
        if row["revoked_at"]:
            raise AuthenticationError("access token has been revoked")
        expiry = _parse_time(row["expires_at"])
        if expiry is not None and expiry <= datetime.now(timezone.utc):
            raise AuthenticationError("access token has expired")
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE app_access_tokens SET last_used_at=? WHERE token_id=?", (now, token_id),
            )
        return SecurityActor(
            row["user_id"], row["display_name"], row["system_role"], token_id,
            tuple(json.loads(row["scopes_json"])), False,
        )

    def revoke_token(
        self, token_id: str, *, actor: SecurityActor | None = None,
    ) -> AccessTokenRecord:
        principal, record = self.actor_or_local(actor), self.get_token(token_id)
        if principal.user_id != record.user_id:
            self.require(principal, Capability.SYSTEM_ADMIN, action="security.token.revoke")
        if (not record.revoked and self.mode is DeploymentMode.SHARED
                and self.get_user(record.user_id).system_role is SystemRole.ADMIN
                and self._viable_admin_count(excluding_token=token_id) == 0):
            raise ValueError("cannot revoke the last viable shared-deployment administrator token")
        if not record.revoked:
            with self.database.transaction() as connection:
                connection.execute(
                    "UPDATE app_access_tokens SET revoked_at=? WHERE token_id=?",
                    (utc_now(), token_id),
                )
        self.audit(principal, "security.token.revoke", "access_token", token_id,
                   outcome="succeeded", details={"owner_id": record.user_id})
        return self.get_token(token_id)

    def capabilities(self, actor: SecurityActor, project_id: str = "") -> frozenset[Capability]:
        if actor.local_session:
            granted = frozenset(ALL_CAPABILITIES)
        else:
            granted = _SYSTEM_CAPABILITIES[actor.system_role]
            if project_id and actor.system_role is not SystemRole.ADMIN:
                row = self.database.fetch_one(
                    "SELECT project_role FROM app_project_memberships "
                    "WHERE project_id=? AND user_id=?", (project_id, actor.user_id),
                )
                if row is not None:
                    granted = granted | _PROJECT_CAPABILITIES[ProjectRole(row["project_role"])]
            granted = granted & frozenset(actor.token_scopes)
        return granted

    def require(
        self, actor: SecurityActor, capability: Capability | str, *, project_id: str = "",
        action: str = "authorize", resource_type: str = "security", resource_id: str = "",
    ) -> None:
        capability = Capability(capability)
        allowed = capability in self.capabilities(actor, project_id)
        self.audit(actor, action, resource_type, resource_id, project_id=project_id,
                   outcome="allowed" if allowed else "denied",
                   details={"required_capability": capability.value})
        if not allowed:
            scope = f" for project {project_id}" if project_id else ""
            raise AuthorizationError(
                f"{actor.user_id} lacks {capability.value} capability{scope}"
            )

    def set_mode(
        self, mode: DeploymentMode | str, *, actor: SecurityActor | None = None,
    ) -> DeploymentMode:
        principal, requested = self.actor_or_local(actor), DeploymentMode(mode)
        self.require(principal, Capability.SYSTEM_ADMIN, action="security.mode.set")
        if requested is DeploymentMode.SHARED and self._viable_admin_count() == 0:
            raise ValueError(
                "shared mode requires an active administrator with a non-expired, "
                "non-revoked token carrying system_admin scope"
            )
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO app_settings(setting_key,value_json,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(setting_key) DO UPDATE SET value_json=excluded.value_json,"
                "updated_at=excluded.updated_at",
                (_MODE_SETTING, _json(requested.value), now),
            )
        self.audit(principal, "security.mode.set", "deployment", requested.value,
                   outcome="succeeded", details={"mode": requested.value})
        return requested

    def _viable_admin_count(self, *, excluding_user: str = "", excluding_token: str = "") -> int:
        now = datetime.now(timezone.utc)
        rows = self.database.fetch_all(
            "SELECT t.*,u.active,u.system_role FROM app_access_tokens t "
            "JOIN app_users u ON u.user_id=t.user_id WHERE u.system_role='admin' "
            "AND u.active=1 AND t.revoked_at=''"
        )
        count = 0
        for row in rows:
            if row["user_id"] == excluding_user or row["token_id"] == excluding_token:
                continue
            if Capability.SYSTEM_ADMIN.value not in json.loads(row["scopes_json"]):
                continue
            expiry = _parse_time(row["expires_at"])
            if expiry is None or expiry > now:
                count += 1
        return count

    @staticmethod
    def _event_digest(payload: dict[str, Any]) -> str:
        return hashlib.sha256(_json(payload).encode()).hexdigest()

    def audit(
        self, actor: SecurityActor | str, action: str, resource_type: str,
        resource_id: str = "", *, project_id: str = "", outcome: str = "succeeded",
        details: dict[str, Any] | None = None,
    ) -> SecurityAuditEvent:
        actor_id = actor.user_id if isinstance(actor, SecurityActor) else nonempty(actor, "actor_id")
        action, resource_type = nonempty(action, "action"), nonempty(resource_type, "resource_type")
        if outcome not in {"allowed", "denied", "succeeded", "failed"}:
            raise ValueError("invalid security audit outcome")
        safe = _safe_details(details or {})
        event_id, created_at = f"audit_{uuid4().hex}", utc_now()
        with self.database.transaction() as connection:
            previous = connection.execute(
                "SELECT sequence,event_hash FROM app_security_audit ORDER BY sequence DESC LIMIT 1"
            ).fetchone()
            sequence = 1 if previous is None else int(previous["sequence"]) + 1
            previous_hash = _GENESIS_HASH if previous is None else previous["event_hash"]
            material = {
                "sequence": sequence, "event_id": event_id, "actor_id": actor_id,
                "action": action, "resource_type": resource_type,
                "resource_id": str(resource_id).strip(), "project_id": str(project_id).strip(),
                "outcome": outcome, "details": safe, "previous_hash": previous_hash,
                "created_at": created_at,
            }
            event_hash = self._event_digest(material)
            connection.execute(
                "INSERT INTO app_security_audit VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (sequence, event_id, actor_id, action, resource_type,
                 material["resource_id"], material["project_id"], outcome, _json(safe),
                 previous_hash, event_hash, created_at),
            )
        return self.audit_event(sequence)

    def audit_event(self, sequence: int) -> SecurityAuditEvent:
        row = self.database.fetch_one(
            "SELECT * FROM app_security_audit WHERE sequence=?", (int(sequence),),
        )
        if row is None:
            raise KeyError(f"unknown security audit sequence: {sequence}")
        return _audit(row)

    def audit_events(
        self, *, project_id: str = "", actor_id: str = "", limit: int = 500,
    ) -> tuple[SecurityAuditEvent, ...]:
        limit = int(limit)
        if not 1 <= limit <= 10_000:
            raise ValueError("audit-event limit must be between 1 and 10000")
        clauses, parameters = [], []
        if project_id:
            clauses.append("project_id=?")
            parameters.append(project_id)
        if actor_id:
            clauses.append("actor_id=?")
            parameters.append(actor_id)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        parameters.append(limit)
        return tuple(_audit(row) for row in self.database.fetch_all(
            "SELECT * FROM app_security_audit" + where + " ORDER BY sequence DESC LIMIT ?",
            tuple(parameters),
        ))

    def verify_audit_chain(self) -> dict[str, Any]:
        previous_hash, expected_sequence = _GENESIS_HASH, 1
        rows = self.database.fetch_all("SELECT * FROM app_security_audit ORDER BY sequence")
        for row in rows:
            event = _audit(row)
            material = {
                "sequence": event.sequence, "event_id": event.event_id,
                "actor_id": event.actor_id, "action": event.action,
                "resource_type": event.resource_type, "resource_id": event.resource_id,
                "project_id": event.project_id, "outcome": event.outcome,
                "details": event.details, "previous_hash": event.previous_hash,
                "created_at": event.created_at,
            }
            if event.sequence != expected_sequence:
                return {"valid": False, "events": len(rows), "failure_sequence": event.sequence,
                        "reason": "non-contiguous sequence"}
            if event.previous_hash != previous_hash:
                return {"valid": False, "events": len(rows), "failure_sequence": event.sequence,
                        "reason": "previous hash mismatch"}
            if not hmac.compare_digest(event.event_hash, self._event_digest(material)):
                return {"valid": False, "events": len(rows), "failure_sequence": event.sequence,
                        "reason": "event hash mismatch"}
            previous_hash, expected_sequence = event.event_hash, expected_sequence + 1
        return {"valid": True, "events": len(rows), "head_hash": previous_hash}


__all__ = [
    "AuthenticationError", "AuthorizationError", "SecurityService",
]
