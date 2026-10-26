from __future__ import annotations

from datetime import datetime, timedelta, timezone
import sqlite3

import pytest

from pymrst.app.application import create_application
from pymrst.app.database.models import APP_DATABASE_SCHEMA_VERSION, AppDatabase
from pymrst.app.schemas.project import ProjectCreate
from pymrst.app.schemas.security import Capability, DeploymentMode
from pymrst.app.services.security_service import AuthenticationError, AuthorizationError


@pytest.fixture
def application(tmp_path):
    context = create_application(tmp_path / "workspace")
    yield context
    context.close()


def _shared(application):
    admin = application.security.create_user("Security Admin", "admin@example.test", "admin")
    issued = application.security.issue_token(
        admin.user_id, "bootstrap", tuple(Capability),
        expires_at=(datetime.now(timezone.utc) + timedelta(days=30)).isoformat(),
    )
    application.security.set_mode(DeploymentMode.SHARED)
    return admin, issued, application.security.authenticate(issued.token)


def test_schema_migrates_to_security_version(tmp_path):
    database = AppDatabase(tmp_path / "metadata.sqlite")
    assert database.schema_version == APP_DATABASE_SCHEMA_VERSION == 20
    tables = {row[0] for row in database.fetch_all(
        "SELECT name FROM sqlite_master WHERE type='table'"
    )}
    assert {"app_users", "app_project_memberships", "app_access_tokens",
            "app_security_audit"} <= tables
    database.close()


def test_local_mode_is_default_and_shared_mode_has_lockout_guard(application):
    assert application.security.mode is DeploymentMode.LOCAL
    user = application.security.create_user("Engineer", "engineer@example.test", "engineer")
    with pytest.raises(ValueError, match="requires an active administrator"):
        application.security.set_mode("shared")
    admin = application.security.create_user("Admin", "admin@example.test", "admin")
    application.security.issue_token(admin.user_id, "insufficient", [Capability.VIEW])
    with pytest.raises(ValueError, match="requires an active administrator"):
        application.security.set_mode("shared")
    assert user.active


def test_token_is_returned_once_and_only_hash_is_persisted(application):
    admin = application.security.create_user("Admin", "admin@example.test", "admin")
    issued = application.security.issue_token(
        admin.user_id, "desktop", [Capability.SYSTEM_ADMIN, Capability.VIEW]
    )
    row = application.database.fetch_one(
        "SELECT * FROM app_access_tokens WHERE token_id=?", (issued.record.token_id,)
    )
    assert issued.token.startswith("pymrst_pat_")
    assert issued.token not in tuple(row)
    assert len(row["token_hash"]) == 64
    assert "token" not in issued.record.to_dict()
    assert application.security.authenticate(issued.token).user_id == admin.user_id
    assert application.security.get_token(issued.record.token_id).last_used_at


def test_project_roles_and_token_scopes_both_constrain_access(application):
    admin, _, actor = _shared(application)
    project = application.projects.create(ProjectCreate("RBAC case"))
    runner = application.security.create_user(
        "Run Engineer", "runner@example.test", "engineer", actor=actor,
    )
    application.security.set_membership(project.project_id, runner.user_id, "runner", actor=actor)
    issued = application.security.issue_token(
        runner.user_id, "run-only", [Capability.VIEW, Capability.EXECUTE], actor=actor,
    )
    run_actor = application.security.authenticate(issued.token)
    application.security.require(run_actor, Capability.EXECUTE, project_id=project.project_id)
    with pytest.raises(AuthorizationError, match="lacks edit"):
        application.security.require(run_actor, Capability.EDIT, project_id=project.project_id)
    with pytest.raises(AuthorizationError, match="lacks execute"):
        application.security.require(run_actor, Capability.EXECUTE, project_id="project_unknown")
    assert admin.system_role.value == "admin"


def test_revocation_expiry_disable_and_last_admin_safety(application):
    admin, issued, actor = _shared(application)
    with pytest.raises(ValueError, match="last viable"):
        application.security.revoke_token(issued.record.token_id, actor=actor)
    second = application.security.issue_token(
        admin.user_id, "second", [Capability.SYSTEM_ADMIN, Capability.VIEW], actor=actor,
    )
    application.security.revoke_token(issued.record.token_id, actor=actor)
    with pytest.raises(AuthenticationError, match="revoked"):
        application.security.authenticate(issued.token)
    second_actor = application.security.authenticate(second.token)
    with pytest.raises(ValueError, match="last viable"):
        application.security.set_user_active(admin.user_id, False, actor=second_actor)
    with application.database.transaction() as connection:
        connection.execute(
            "UPDATE app_access_tokens SET expires_at=? WHERE token_id=?",
            ((datetime.now(timezone.utc) - timedelta(seconds=1)).isoformat(),
             second.record.token_id),
        )
    with pytest.raises(AuthenticationError, match="expired"):
        application.security.authenticate(second.token)


def test_shared_mode_rejects_local_identity_and_missing_identity(application):
    _, _, _ = _shared(application)
    with pytest.raises(AuthenticationError, match="requires a bearer token"):
        application.security.actor_or_local(None)
    with pytest.raises(AuthenticationError, match="disabled in shared mode"):
        application.security.actor_or_local(application.security.local_actor())


def test_audit_log_is_hash_chained_and_detects_tampering(application):
    application.security.create_user("Admin", "admin@example.test", "admin")
    verified = application.security.verify_audit_chain()
    assert verified["valid"] is True
    assert verified["events"] >= 2
    events = tuple(reversed(application.security.audit_events()))
    assert events[0].previous_hash == "0" * 64
    assert events[-1].previous_hash == events[-2].event_hash
    with application.database.transaction() as connection:
        connection.execute(
            "UPDATE app_security_audit SET action='tampered' WHERE sequence=?",
            (events[-1].sequence,),
        )
    failure = application.security.verify_audit_chain()
    assert failure["valid"] is False
    assert failure["failure_sequence"] == events[-1].sequence
    assert failure["reason"] == "event hash mismatch"


def test_audit_rejects_sensitive_details(application):
    with pytest.raises(ValueError, match="sensitive field"):
        application.security.audit(
            "operator", "test", "test", details={"api_token": "do-not-store"}
        )


def test_existing_version_17_database_upgrades_without_data_loss(tmp_path):
    path = tmp_path / "legacy.sqlite"
    database = AppDatabase(path)
    database.close()
    connection = sqlite3.connect(path)
    connection.execute("DROP TABLE app_security_audit")
    connection.execute("DROP TABLE app_access_tokens")
    connection.execute("DROP TABLE app_project_memberships")
    connection.execute("DROP TABLE app_users")
    connection.execute("PRAGMA user_version=17")
    connection.commit()
    connection.close()
    upgraded = AppDatabase(path)
    assert upgraded.schema_version == 20
    assert upgraded.fetch_one(
        "SELECT name FROM sqlite_master WHERE name='app_users'"
    ) is not None
    upgraded.close()
