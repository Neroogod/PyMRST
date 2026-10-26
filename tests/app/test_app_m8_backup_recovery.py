"""Production-hardening acceptance tests for backup and staged recovery."""

from pathlib import Path
import sqlite3
import time

import pytest

from pymrst.app import create_application
from pymrst.app.database import AppDatabase, APP_DATABASE_SCHEMA_VERSION
from pymrst.app.schemas import ProjectCreate, WorkspaceState
from pymrst.app.services.backup_service import BackupCancelled


def test_workspace_backup_is_checksummed_verified_and_restorable(tmp_path):
    source = tmp_path / "SPE1.DATA"
    source.write_text("RUNSPEC\nDIMENS\n10 10 3 /\n", encoding="utf-8")
    workspace = tmp_path / "workspace"
    restored_root = tmp_path / "restored-workspace"
    with create_application(workspace) as app:
        project = app.projects.create(ProjectCreate("Recovery case"))
        asset = app.projects.import_input(project.project_id, source)
        app.state.save_workspace(WorkspaceState(
            active_project_id=project.project_id,
            route=f"/projects/{project.project_id}/model",
        ))
        plan = app.backups.plan()
        assert plan.project_count == 1
        assert plan.file_count >= 3
        assert plan.database_schema_version == app.database.schema_version
        backup = app.backups.create()
        archive = workspace / backup.relative_path
        assert archive.is_file() and archive.stat().st_size == backup.size_bytes
        verified = app.backups.verify(backup.backup_id)
        assert verified.status == "verified"
        assert verified.verified_at
        assert verified.manifest["file_count"] == backup.file_count
        receipt = app.backups.restore(backup.backup_id, restored_root)
        assert receipt.database_integrity == "ok"
        assert receipt.archive_hash == backup.content_hash

    with create_application(restored_root) as restored:
        recovered = restored.projects.get(project.project_id)
        assert Path(recovered.root_path).parent == restored_root / "projects"
        assert restored.projects.asset_path(project.project_id, asset.asset_id).read_text() \
            == source.read_text()
        assert restored.state.workspace().active_project_id == project.project_id
        connection = sqlite3.connect(restored.database.path)
        try:
            assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        finally:
            connection.close()


def test_restore_refuses_live_or_existing_target_and_archive_tampering(tmp_path):
    workspace = tmp_path / "workspace"
    with create_application(workspace) as app:
        app.projects.create(ProjectCreate("Safety case"))
        backup = app.backups.create()
        with pytest.raises(ValueError, match="live workspace"):
            app.backups.restore(backup.backup_id, workspace)
        with pytest.raises(ValueError, match="live workspace"):
            app.backups.restore(backup.backup_id, workspace / "nested-restore")
        existing = tmp_path / "already-there"
        existing.mkdir()
        with pytest.raises(FileExistsError, match="must not already exist"):
            app.backups.restore(backup.backup_id, existing)

        archive = workspace / backup.relative_path
        archive.chmod(0o644)
        with archive.open("ab") as stream:
            stream.write(b"tampered")
        with pytest.raises(RuntimeError, match="outer SHA-256"):
            app.backups.verify(backup.backup_id)
        assert app.backups.get(backup.backup_id).status == "failed"


def test_backup_refuses_project_symlinks(tmp_path):
    workspace = tmp_path / "workspace"
    with create_application(workspace) as app:
        project = app.projects.create(ProjectCreate("Symlink safety"))
        source = tmp_path / "outside.txt"
        source.write_text("not part of the project")
        link = Path(project.root_path) / "models" / "external-link"
        try:
            link.symlink_to(source)
        except OSError:
            pytest.skip("filesystem does not permit symlink creation")
        with pytest.raises(RuntimeError, match="refuses project symlink"):
            app.backups.plan()


def test_schema_15_database_migrates_forward_to_backup_registry(tmp_path):
    path = tmp_path / "metadata.sqlite"
    database = AppDatabase(path)
    with database.transaction() as connection:
        connection.execute("DROP TABLE app_backup_qualifications")
        connection.execute("DROP TABLE app_backup_jobs")
        connection.execute("DROP TABLE app_workspace_backups")
        connection.execute("PRAGMA user_version=15")
    database.close()
    migrated = AppDatabase(path)
    try:
        assert migrated.schema_version == APP_DATABASE_SCHEMA_VERSION == 20
        assert migrated.fetch_one(
            "SELECT name FROM sqlite_master WHERE type='table' "
            "AND name='app_workspace_backups'"
        )["name"] == "app_workspace_backups"
    finally:
        migrated.close()


def wait_for_backup_job(app, job_id, timeout=5):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = app.backups.job(job_id)
        if job.status.value in {"completed", "failed", "cancelled", "interrupted"}:
            return job
        time.sleep(0.01)
    raise TimeoutError(job_id)


def test_nonblocking_job_verifies_and_qualifies_recovery(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        app.projects.create(ProjectCreate("Background backup"))
        queued = app.backups.submit()
        assert queued.status.value in {"queued", "running"}
        completed = wait_for_backup_job(app, queued.job_id)
        assert completed.status.value == "completed"
        assert completed.progress == 1
        backup = app.backups.get(completed.backup_id)
        assert backup.status == "verified"
        qualification = app.backups.qualify(backup.backup_id)
        assert qualification.status == "passed"
        assert qualification.checks["database_integrity"] == "ok"
        assert qualification.checks["disposable_target_removed"] is True


def test_disabled_schedule_is_inert_and_enabled_retention_is_auditable(tmp_path):
    workspace = tmp_path / "workspace"
    with create_application(workspace) as app:
        app.projects.create(ProjectCreate("Retention policy"))
        assert app.backups.schedule_if_due() is None
        created = []
        for _ in range(3):
            created.append(app.backups.verify(app.backups.create().backup_id))
        policy_type = type(app.backups.policy())
        app.backups.set_policy(policy_type(
            enabled=True, interval_hours=24, retained_verified_backups=1,
            auto_verify=True, recovery_drill=False,
        ))
        pruned = app.backups.apply_retention()
        assert len(pruned) == 2
        assert all(item.pruned_at for item in pruned)
        assert all(not (workspace / item.relative_path).exists() for item in pruned)
        retained = [item for item in app.backups.list() if not item.pruned_at]
        assert [item.backup_id for item in retained] == [created[-1].backup_id]


def test_restart_marks_unfinished_backup_job_interrupted(tmp_path):
    root = tmp_path / "workspace"
    with create_application(root) as app:
        now = "2026-01-01T00:00:00+00:00"
        with app.database.transaction() as connection:
            connection.execute(
                "INSERT INTO app_backup_jobs VALUES(?,?,?,?,?,?,?,?,?)",
                ("backup_job_abandoned", "running", .4, "Streaming", None,
                 None, 0, now, now),
            )
    with create_application(root) as reopened:
        recovered = reopened.backups.job("backup_job_abandoned")
        assert recovered.status.value == "interrupted"
        assert "restarted" in recovered.message.lower()


def test_background_backup_cancellation_is_cooperative_and_leaves_no_archive(
    tmp_path, monkeypatch,
):
    with create_application(tmp_path / "workspace") as app:
        app.projects.create(ProjectCreate("Cancellation"))

        def blocking_create(*, progress, cancelled):
            progress(.25, "Waiting at safe cancellation point")
            deadline = time.monotonic() + 2
            while time.monotonic() < deadline and not cancelled():
                time.sleep(.005)
            raise BackupCancelled("backup cancellation requested")

        monkeypatch.setattr(app.backups, "_create_unlocked", blocking_create)
        submitted = app.backups.submit()
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            running = app.backups.job(submitted.job_id)
            if running.status.value == "running":
                break
            time.sleep(.005)
        app.backups.cancel(submitted.job_id)
        cancelled = wait_for_backup_job(app, submitted.job_id)
        assert cancelled.status.value == "cancelled"
        assert app.backups.list() == ()
        assert not tuple((tmp_path / "workspace" / ".pymrst-app" / "backups").glob("*.zip"))


def test_enabled_schedule_runs_verified_backup_and_recovery_drill(tmp_path):
    with create_application(tmp_path / "workspace") as app:
        app.projects.create(ProjectCreate("Scheduled recovery"))
        policy_type = type(app.backups.policy())
        app.backups.set_policy(policy_type(
            enabled=True, interval_hours=1, retained_verified_backups=2,
            auto_verify=True, recovery_drill=True,
        ))
        scheduled = app.backups.schedule_if_due()
        assert scheduled is not None
        completed = wait_for_backup_job(app, scheduled.job_id)
        assert completed.status.value == "completed"
        assert app.backups.get(completed.backup_id).status == "verified"
        qualifications = app.backups.qualifications(completed.backup_id)
        assert len(qualifications) == 1 and qualifications[0].status == "passed"
        assert app.backups.schedule_if_due() is None
