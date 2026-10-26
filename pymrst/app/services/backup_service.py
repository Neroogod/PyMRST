"""Bounded, checksummed workspace backup and staged disaster recovery."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import tempfile
import os
import threading
import time
from uuid import uuid4
import zipfile

from ..database.models import APP_DATABASE_SCHEMA_VERSION, AppDatabase
from ..schemas._common import utc_now
from ..schemas.backup import (
    BackupJobRecord, BackupJobStatus, BackupPlan, BackupPolicy,
    BackupQualificationRecord, BackupRecord, RestoreReceipt,
)
from .project_service import ProjectService


MANIFEST_PATH = ".pymrst-backup/manifest.json"
DATABASE_PATH = ".pymrst-app/metadata.sqlite"
BACKUP_POLICY_KEY = "workspace_backup_policy"


class BackupCancelled(RuntimeError):
    pass


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _record(row) -> BackupRecord:
    return BackupRecord(
        row["backup_id"], row["relative_path"], row["content_hash"],
        row["size_bytes"], row["project_count"], row["file_count"],
        row["status"], json.loads(row["manifest_json"]), row["created_at"],
        row["verified_at"], row["pruned_at"],
    )


def _job(row) -> BackupJobRecord:
    return BackupJobRecord(
        row["job_id"], row["status"], row["progress"], row["message"],
        row["backup_id"] or "",
        None if row["error_json"] is None else json.loads(row["error_json"]),
        bool(row["cancellation_requested"]), row["created_at"], row["updated_at"],
    )


def _qualification(row) -> BackupQualificationRecord:
    return BackupQualificationRecord(
        row["qualification_id"], row["backup_id"], row["status"],
        json.loads(row["checks_json"]), row["created_at"],
    )


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (not name or name.endswith("/") or path.is_absolute() or ".." in path.parts
            or "\\" in name or str(path) != name):
        raise ValueError(f"unsafe backup archive member: {name!r}")
    return path


def _write_member(
    archive: zipfile.ZipFile, archive_name: str, source: Path,
    *, on_bytes=None, cancelled=None,
):
    """Write and hash exactly the same byte stream, detecting live changes."""
    digest = hashlib.sha256()
    with source.open("rb") as reader:
        before = os.fstat(reader.fileno())
        with archive.open(archive_name, "w", force_zip64=True) as writer:
            for block in iter(lambda: reader.read(1024 * 1024), b""):
                if cancelled is not None and cancelled():
                    raise BackupCancelled("backup cancellation requested")
                writer.write(block)
                digest.update(block)
                if on_bytes is not None:
                    on_bytes(len(block))
        after = os.fstat(reader.fileno())
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError(f"file changed while backup was created: {source}")
    return before.st_size, digest.hexdigest()


class WorkspaceBackupService:
    """Create complete workspace snapshots without pausing active readers.

    Simulations remain isolated and untouched. SQLite's online backup API
    captures a consistent metadata snapshot while project files are streamed
    into the archive with independent SHA-256 checksums.
    """

    def __init__(
        self, database: AppDatabase, projects: ProjectService, workspace_root: str | Path,
        *, maximum_files: int = 1_000_000,
        maximum_uncompressed_bytes: int = 2 * 1024**4,
        maximum_member_bytes: int = 100 * 1024**3,
    ):
        self.database, self.projects = database, projects
        self.workspace_root = Path(workspace_root).resolve()
        self.backups_root = self.workspace_root / ".pymrst-app" / "backups"
        self.backups_root.mkdir(parents=True, exist_ok=True)
        self.maximum_files = int(maximum_files)
        self.maximum_uncompressed_bytes = int(maximum_uncompressed_bytes)
        self.maximum_member_bytes = int(maximum_member_bytes)
        self._creation_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="pymrst-backup",
        )
        self._cancellations: dict[str, threading.Event] = {}
        self._job_lock = threading.RLock()
        self._scheduler_stop = threading.Event()
        self._scheduler_thread: threading.Thread | None = None
        if min(self.maximum_files, self.maximum_uncompressed_bytes,
               self.maximum_member_bytes) <= 0:
            raise ValueError("backup resource bounds must be positive")

    def _project_files(self):
        output = []
        for project in self.projects.list():
            root = Path(project.root_path).resolve()
            expected = (self.projects.projects_root / project.project_id).resolve()
            if root != expected or root.is_symlink() or not root.is_dir():
                raise RuntimeError(f"project root is unsafe or missing: {project.project_id}")
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    raise RuntimeError(f"backup refuses project symlink: {path}")
                if path.is_file():
                    relative = path.relative_to(self.workspace_root).as_posix()
                    output.append((relative, path, path.stat().st_size))
        if len(output) + 1 > self.maximum_files:
            raise ValueError("workspace exceeds configured backup file-count limit")
        total = sum(item[2] for item in output) + self.database.path.stat().st_size
        if total > self.maximum_uncompressed_bytes:
            raise ValueError("workspace exceeds configured backup size limit")
        return tuple(output), total

    def plan(self) -> BackupPlan:
        files, total = self._project_files()
        return BackupPlan(
            len(self.projects.list()), len(files) + 1, total,
            self.database.schema_version,
            (DATABASE_PATH, "projects/"),
        )

    def create(self, *, progress=None, cancelled=None) -> BackupRecord:
        with self._creation_lock:
            return self._create_unlocked(progress=progress, cancelled=cancelled)

    def _create_unlocked(self, *, progress=None, cancelled=None) -> BackupRecord:
        progress = progress or (lambda _value, _message: None)
        cancelled = cancelled or (lambda: False)
        if cancelled():
            raise BackupCancelled("backup cancellation requested")
        progress(0.01, "Inventorying workspace")
        plan = self.plan()
        backup_id = f"backup_{uuid4().hex}"
        relative = Path(".pymrst-app/backups") / f"{backup_id}.pymrst-backup.zip"
        target = self.workspace_root / relative
        temporary = target.with_suffix(target.suffix + f".tmp-{uuid4().hex}")
        snapshot = self.backups_root / f".{backup_id}.metadata.sqlite"
        created_at = utc_now()
        files, _ = self._project_files()
        entries = []
        completed_bytes = 0

        def advance(count):
            nonlocal completed_bytes
            completed_bytes += count
            denominator = max(1, plan.estimated_bytes)
            progress(min(0.88, 0.08 + 0.80 * completed_bytes / denominator),
                     "Streaming checksummed project data")

        try:
            if cancelled():
                raise BackupCancelled("backup cancellation requested")
            progress(0.04, "Creating consistent metadata snapshot")
            self.database.backup_to(snapshot)
            sources = ((DATABASE_PATH, snapshot, snapshot.stat().st_size), *files)
            total = 0
            with zipfile.ZipFile(
                temporary, "w", compression=zipfile.ZIP_DEFLATED,
                compresslevel=6, allowZip64=True,
            ) as archive:
                for archive_name, source, expected_size in sources:
                    if cancelled():
                        raise BackupCancelled("backup cancellation requested")
                    _safe_member(archive_name)
                    size = source.stat().st_size
                    if size != expected_size and source is not snapshot:
                        raise RuntimeError(f"file changed while backup was created: {source}")
                    if size > self.maximum_member_bytes:
                        raise ValueError(f"backup member exceeds size limit: {archive_name}")
                    total += size
                    if total > self.maximum_uncompressed_bytes:
                        raise ValueError("workspace changed beyond configured backup size limit")
                    written_size, digest = _write_member(
                        archive, archive_name, source, on_bytes=advance,
                        cancelled=cancelled,
                    )
                    if written_size != size:
                        raise RuntimeError(f"file changed while backup was created: {source}")
                    entries.append({
                        "path": archive_name, "size_bytes": written_size, "sha256": digest,
                    })
                manifest = {
                    "schema_version": 1, "backup_id": backup_id,
                    "created_at": created_at,
                    "source_workspace_root": str(self.workspace_root),
                    "database_schema_version": APP_DATABASE_SCHEMA_VERSION,
                    "project_ids": [item.project_id for item in self.projects.list()],
                    "file_count": len(entries), "uncompressed_bytes": total,
                    "files": entries,
                }
                archive.writestr(
                    MANIFEST_PATH,
                    json.dumps(manifest, sort_keys=True, separators=(",", ":"),
                               ensure_ascii=False, allow_nan=False) + "\n",
                )
            if cancelled():
                raise BackupCancelled("backup cancellation requested")
            progress(0.90, "Sealing archive identity")
            digest, size = _hash_file(temporary), temporary.stat().st_size
            temporary.replace(target)
            target.chmod(0o444)
            with self.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO app_workspace_backups "
                    "(backup_id,relative_path,content_hash,size_bytes,project_count,"
                    "file_count,status,manifest_json,created_at,verified_at,pruned_at) "
                    "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (backup_id, relative.as_posix(), digest, size,
                     plan.project_count, len(entries), "created",
                     json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                     created_at, "", ""),
                )
            progress(0.94, "Archive recorded")
            return self.get(backup_id)
        finally:
            snapshot.unlink(missing_ok=True)
            temporary.unlink(missing_ok=True)

    def get(self, backup_id: str) -> BackupRecord:
        row = self.database.fetch_one(
            "SELECT * FROM app_workspace_backups WHERE backup_id=?", (str(backup_id),),
        )
        if row is None:
            raise KeyError(f"unknown workspace backup: {backup_id}")
        return _record(row)

    def list(self):
        return tuple(_record(row) for row in self.database.fetch_all(
            "SELECT * FROM app_workspace_backups ORDER BY created_at, backup_id",
        ))

    def _archive_path(self, record: BackupRecord) -> Path:
        if record.pruned_at:
            raise RuntimeError("backup archive was pruned by the retention policy")
        target = (self.workspace_root / record.relative_path).resolve()
        if self.backups_root.resolve() not in target.parents or target.is_symlink():
            raise RuntimeError("backup archive path is outside the managed backup root")
        if not target.is_file():
            raise RuntimeError("backup archive is missing")
        return target

    def _verify_archive(self, record: BackupRecord):
        path = self._archive_path(record)
        if _hash_file(path) != record.content_hash:
            raise RuntimeError("backup archive failed its outer SHA-256 checksum")
        with zipfile.ZipFile(path, "r") as archive:
            infos = archive.infolist()
            names = [item.filename for item in infos]
            if len(names) != len(set(names)):
                raise RuntimeError("backup archive contains duplicate paths")
            if MANIFEST_PATH not in names:
                raise RuntimeError("backup archive has no manifest")
            for name in names:
                _safe_member(name)
            try:
                manifest = json.loads(archive.read(MANIFEST_PATH))
            except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise RuntimeError("backup manifest is unreadable") from error
            required = {
                "schema_version", "backup_id", "created_at", "source_workspace_root",
                "database_schema_version", "project_ids", "file_count",
                "uncompressed_bytes", "files",
            }
            if set(manifest) != required or manifest["schema_version"] != 1:
                raise RuntimeError("backup manifest contract is unsupported")
            if manifest["backup_id"] != record.backup_id:
                raise RuntimeError("backup manifest identity mismatch")
            entries = manifest["files"]
            if not isinstance(entries, list) or len(entries) > self.maximum_files:
                raise RuntimeError("backup manifest file count exceeds safe bounds")
            by_name = {item.filename: item for item in infos}
            expected_names = {MANIFEST_PATH}
            total = 0
            for entry in entries:
                if set(entry) != {"path", "size_bytes", "sha256"}:
                    raise RuntimeError("backup manifest file entry is invalid")
                name = str(_safe_member(entry["path"]))
                if name != DATABASE_PATH and not name.startswith("projects/"):
                    raise RuntimeError(f"backup contains an unsupported member: {name}")
                if name in expected_names or name not in by_name:
                    raise RuntimeError("backup manifest member set is inconsistent")
                expected_names.add(name)
                size = int(entry["size_bytes"])
                if size < 0 or size > self.maximum_member_bytes:
                    raise RuntimeError("backup member size exceeds safe bounds")
                info = by_name[name]
                if info.file_size != size:
                    raise RuntimeError(f"backup member size mismatch: {name}")
                total += size
                if total > self.maximum_uncompressed_bytes:
                    raise RuntimeError("backup uncompressed size exceeds safe bounds")
                digest = hashlib.sha256()
                with archive.open(info, "r") as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b""):
                        digest.update(block)
                if digest.hexdigest() != entry["sha256"]:
                    raise RuntimeError(f"backup member checksum mismatch: {name}")
            if set(names) != expected_names or total != int(manifest["uncompressed_bytes"]):
                raise RuntimeError("backup manifest does not cover the complete archive")
            return manifest

    def verify(self, backup_id: str) -> BackupRecord:
        record = self.get(backup_id)
        try:
            manifest = self._verify_archive(record)
        except Exception:
            with self.database.transaction() as connection:
                connection.execute(
                    "UPDATE app_workspace_backups SET status='failed',verified_at=? "
                    "WHERE backup_id=?", (utc_now(), backup_id),
                )
            raise
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE app_workspace_backups SET status='verified',manifest_json=?,"
                "verified_at=? WHERE backup_id=?",
                (json.dumps(manifest, sort_keys=True, separators=(",", ":")), now, backup_id),
            )
        return self.get(backup_id)

    def restore(self, backup_id: str, target_root: str | Path) -> RestoreReceipt:
        """Restore into a new workspace; never overwrite the live workspace."""
        record = self.verify(backup_id)
        target = Path(target_root).resolve()
        if (target == self.workspace_root or target in self.workspace_root.parents
                or self.workspace_root in target.parents):
            raise ValueError(
                "restore target cannot be the live workspace, its parent, or its child"
            )
        if target.exists():
            raise FileExistsError("restore target must not already exist")
        target.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.restore-", dir=target.parent))
        restored_bytes = 0
        try:
            archive_path = self._archive_path(record)
            manifest = record.manifest
            with zipfile.ZipFile(archive_path, "r") as archive:
                for entry in manifest["files"]:
                    relative = _safe_member(entry["path"])
                    destination = staging.joinpath(*relative.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(entry["path"], "r") as source, destination.open("xb") as sink:
                        shutil.copyfileobj(source, sink, length=1024 * 1024)
                    restored_bytes += destination.stat().st_size
            if _hash_file(archive_path) != record.content_hash:
                raise RuntimeError("backup archive changed during staged recovery")
            database_path = staging / DATABASE_PATH
            connection = sqlite3.connect(database_path)
            try:
                source_root = Path(manifest["source_workspace_root"]).resolve()
                rows = connection.execute(
                    "SELECT project_id,root_path FROM app_projects"
                ).fetchall()
                for project_id, root_path in rows:
                    expected = (source_root / "projects" / project_id).resolve()
                    if Path(root_path).resolve() != expected:
                        raise RuntimeError(f"backup contains unsafe project root: {project_id}")
                    connection.execute(
                        "UPDATE app_projects SET root_path=? WHERE project_id=?",
                        (str(target / "projects" / project_id), project_id),
                    )
                connection.commit()
                integrity = connection.execute("PRAGMA integrity_check").fetchone()[0]
                foreign_keys = connection.execute("PRAGMA foreign_key_check").fetchall()
                if integrity != "ok" or foreign_keys:
                    raise RuntimeError("restored metadata database failed integrity checks")
            finally:
                connection.close()
            receipt = RestoreReceipt(
                backup_id, str(target), record.content_hash, len(manifest["files"]),
                restored_bytes, integrity, utc_now(),
            )
            receipt_path = staging / ".pymrst-app" / "restore-receipt.json"
            receipt_path.write_text(
                json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            staging.replace(target)
            return receipt
        finally:
            if staging.exists():
                shutil.rmtree(staging)

    def policy(self) -> BackupPolicy:
        row = self.database.fetch_one(
            "SELECT value_json FROM app_settings WHERE setting_key=?",
            (BACKUP_POLICY_KEY,),
        )
        return BackupPolicy() if row is None else BackupPolicy.from_dict(
            json.loads(row["value_json"])
        )

    def set_policy(self, policy: BackupPolicy) -> BackupPolicy:
        if not isinstance(policy, BackupPolicy):
            raise TypeError("backup policy update requires BackupPolicy")
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO app_settings(setting_key,value_json,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(setting_key) DO UPDATE SET "
                "value_json=excluded.value_json,updated_at=excluded.updated_at",
                (BACKUP_POLICY_KEY, json.dumps(
                    policy.to_dict(), sort_keys=True, separators=(",", ":")
                ), now),
            )
        return self.policy()

    def job(self, job_id: str) -> BackupJobRecord:
        row = self.database.fetch_one(
            "SELECT * FROM app_backup_jobs WHERE job_id=?", (str(job_id),),
        )
        if row is None:
            raise KeyError(f"unknown backup job: {job_id}")
        return _job(row)

    def jobs(self):
        return tuple(_job(row) for row in self.database.fetch_all(
            "SELECT * FROM app_backup_jobs ORDER BY created_at,job_id",
        ))

    def submit(self) -> BackupJobRecord:
        with self._job_lock:
            active = self.database.fetch_one(
                "SELECT job_id FROM app_backup_jobs WHERE status IN ('queued','running')"
            )
            if active is not None:
                raise ValueError(f"backup job already active: {active['job_id']}")
            job_id, now = f"backup_job_{uuid4().hex}", utc_now()
            with self.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO app_backup_jobs VALUES(?,?,?,?,?,?,?,?,?)",
                    (job_id, "queued", 0.0, "Queued", None, None, 0, now, now),
                )
            event = threading.Event()
            self._cancellations[job_id] = event
            self._executor.submit(self._run_job, job_id, event)
        return self.job(job_id)

    def _update_job(
        self, job_id: str, *, status=None, progress=None, message=None,
        backup_id=None, error=None,
    ):
        current = self.job(job_id)
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE app_backup_jobs SET status=?,progress=?,message=?,backup_id=?,"
                "error_json=?,updated_at=? WHERE job_id=?",
                (
                    current.status.value if status is None else BackupJobStatus(status).value,
                    current.progress if progress is None else float(progress),
                    current.message if message is None else str(message),
                    (current.backup_id or None) if backup_id is None else (backup_id or None),
                    (None if current.error is None else json.dumps(current.error))
                    if error is None else json.dumps(error),
                    utc_now(), job_id,
                ),
            )
        return self.job(job_id)

    def _run_job(self, job_id: str, cancellation: threading.Event):
        last_progress, last_update = 0.0, 0.0

        def report(value, message):
            nonlocal last_progress, last_update
            value, now = max(last_progress, min(float(value), 0.99)), time.monotonic()
            if value - last_progress >= 0.01 or now - last_update >= 0.5:
                self._update_job(
                    job_id, progress=value, message=message,
                )
                last_progress, last_update = value, now

        try:
            current = self.job(job_id)
            if current.cancellation_requested or cancellation.is_set():
                self._update_job(
                    job_id, status="cancelled", message="Cancelled before start",
                )
                return
            self._update_job(job_id, status="running", message="Starting backup")
            backup = self.create(
                progress=report,
                cancelled=cancellation.is_set,
            )
            policy = self.policy()
            if policy.auto_verify:
                report(0.95, "Verifying complete archive")
                backup = self.verify(backup.backup_id)
            if policy.recovery_drill:
                report(0.97, "Running disposable recovery drill")
                self.qualify(backup.backup_id)
            if policy.enabled:
                self.apply_retention()
            self._update_job(
                job_id, status="completed", progress=1.0,
                message="Backup completed", backup_id=backup.backup_id,
            )
        except BackupCancelled:
            self._update_job(
                job_id, status="cancelled", message="Backup cancelled safely",
            )
        except BaseException as error:
            self._update_job(
                job_id, status="failed", message="Backup failed",
                error={"error_type": type(error).__name__, "message": str(error)},
            )
        finally:
            with self._job_lock:
                self._cancellations.pop(job_id, None)

    def cancel(self, job_id: str) -> BackupJobRecord:
        current = self.job(job_id)
        if current.status not in {BackupJobStatus.QUEUED, BackupJobStatus.RUNNING}:
            raise ValueError("only queued or running backup jobs can be cancelled")
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE app_backup_jobs SET cancellation_requested=1,updated_at=? "
                "WHERE job_id=?", (utc_now(), job_id),
            )
        with self._job_lock:
            event = self._cancellations.get(job_id)
            if event is not None:
                event.set()
        return self.job(job_id)

    def recover_interrupted(self):
        rows = self.database.fetch_all(
            "SELECT job_id FROM app_backup_jobs WHERE status IN ('queued','running')"
        )
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE app_backup_jobs SET status='interrupted',"
                "message='Application restarted during backup',updated_at=? "
                "WHERE status IN ('queued','running')", (now,),
            )
        return tuple(self.job(row["job_id"]) for row in rows)

    def qualify(self, backup_id: str) -> BackupQualificationRecord:
        self.get(backup_id)
        qualification_id = f"backup_qualification_{uuid4().hex}"
        created_at, status, checks = utc_now(), "passed", {}
        try:
            with tempfile.TemporaryDirectory(prefix="pymrst-recovery-drill-") as directory:
                target = Path(directory) / "restored-workspace"
                receipt = self.restore(backup_id, target)
                database = target / DATABASE_PATH
                checks = {
                    "archive_verified": True,
                    "database_integrity": receipt.database_integrity,
                    "restored_file_count": receipt.file_count,
                    "restored_bytes": receipt.restored_bytes,
                    "project_manifest_count": len(self.get(backup_id).manifest["project_ids"]),
                }
                if not database.is_file():
                    raise RuntimeError("recovery drill produced no metadata database")
            checks["disposable_target_removed"] = not Path(directory).exists()
        except Exception as error:
            status = "failed"
            checks = {"error_type": type(error).__name__, "error": str(error)}
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO app_backup_qualifications VALUES(?,?,?,?,?)",
                (qualification_id, backup_id, status,
                 json.dumps(checks, sort_keys=True, separators=(",", ":")), created_at),
            )
        record = self.qualification(qualification_id)
        if status == "failed":
            raise RuntimeError(f"backup recovery drill failed: {checks['error']}")
        return record

    def qualification(self, qualification_id: str):
        row = self.database.fetch_one(
            "SELECT * FROM app_backup_qualifications WHERE qualification_id=?",
            (str(qualification_id),),
        )
        if row is None:
            raise KeyError(f"unknown backup qualification: {qualification_id}")
        return _qualification(row)

    def qualifications(self, backup_id: str | None = None):
        if backup_id is None:
            rows = self.database.fetch_all(
                "SELECT * FROM app_backup_qualifications ORDER BY created_at"
            )
        else:
            self.get(backup_id)
            rows = self.database.fetch_all(
                "SELECT * FROM app_backup_qualifications WHERE backup_id=? "
                "ORDER BY created_at", (backup_id,),
            )
        return tuple(_qualification(row) for row in rows)

    def apply_retention(self):
        policy = self.policy()
        if not policy.enabled:
            raise ValueError("automatic backup retention is disabled")
        candidates = [item for item in reversed(self.list())
                      if item.status == "verified" and not item.pruned_at]
        pruned = []
        for record in candidates[policy.retained_verified_backups:]:
            target = self._archive_path(record)
            if _hash_file(target) != record.content_hash:
                raise RuntimeError(
                    f"retention refused checksum-mismatched archive: {record.backup_id}"
                )
            target.chmod(0o600)
            target.unlink()
            now = utc_now()
            with self.database.transaction() as connection:
                connection.execute(
                    "UPDATE app_workspace_backups SET pruned_at=? WHERE backup_id=?",
                    (now, record.backup_id),
                )
            pruned.append(self.get(record.backup_id))
        return tuple(pruned)

    def schedule_if_due(self):
        policy = self.policy()
        if not policy.enabled:
            return None
        active = self.database.fetch_one(
            "SELECT 1 FROM app_backup_jobs WHERE status IN ('queued','running')"
        )
        if active is not None:
            return None
        latest = self.database.fetch_one(
            "SELECT updated_at FROM app_backup_jobs WHERE status='completed' "
            "ORDER BY updated_at DESC LIMIT 1"
        )
        if latest is not None:
            last = datetime.fromisoformat(latest["updated_at"])
            if last.tzinfo is None:
                last = last.replace(tzinfo=timezone.utc)
            if datetime.now(timezone.utc) < last + timedelta(hours=policy.interval_hours):
                return None
        return self.submit()

    def start_scheduler(self, *, poll_seconds: float = 60.0):
        poll_seconds = float(poll_seconds)
        if poll_seconds <= 0:
            raise ValueError("backup scheduler poll interval must be positive")
        if self._scheduler_thread is not None:
            return

        def run():
            while not self._scheduler_stop.wait(poll_seconds):
                try:
                    self.schedule_if_due()
                except Exception:
                    # Scheduler failures are retained by jobs; polling must not
                    # terminate the host application.
                    continue

        self._scheduler_thread = threading.Thread(
            target=run, name="pymrst-backup-scheduler", daemon=True,
        )
        self._scheduler_thread.start()

    def shutdown(self, *, wait=True):
        self._scheduler_stop.set()
        if self._scheduler_thread is not None:
            self._scheduler_thread.join(timeout=2.0)
        with self._job_lock:
            for event in self._cancellations.values():
                event.set()
        self._executor.shutdown(wait=wait, cancel_futures=False)


__all__ = ["BackupCancelled", "WorkspaceBackupService"]
