"""Durable configuration, run, event, and checkpoint metadata services."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

from ..database.models import AppDatabase
from ..schemas._common import json_value, nonempty, utc_now
from ..schemas.run import RunCheckpoint, RunConfiguration, RunEvent, RunRecord, RunStatus
from .project_service import ProjectService, _file_hash
from .task_service import TaskService


def _configuration(row) -> RunConfiguration:
    return RunConfiguration(
        row["configuration_id"], row["project_id"], row["model_id"] or "",
        row["name"], row["runner_kind"], row["configuration_hash"],
        json.loads(row["payload_json"]), row["created_at"],
    )


def _run(row) -> RunRecord:
    return RunRecord(
        row["run_id"], row["project_id"], row["configuration_id"], row["task_id"],
        row["attempt_number"], row["parent_run_id"] or "",
        row["resume_checkpoint_id"], row["status"], row["run_relative_path"],
        row["process_id"], row["started_at"], row["finished_at"], row["created_at"],
    )


class RunService:
    def __init__(self, database: AppDatabase, projects: ProjectService, tasks: TaskService) -> None:
        self.database, self.projects, self.tasks = database, projects, tasks

    def create_configuration(
        self, project_id: str, name: str, runner_kind: str, payload,
        *, model_id: str = "",
    ) -> RunConfiguration:
        self.projects.get(project_id)
        name, runner_kind = nonempty(name, "configuration name"), nonempty(runner_kind, "runner kind")
        if model_id and self.database.fetch_one(
            "SELECT 1 FROM app_models WHERE model_id=? AND project_id=?", (model_id, project_id),
        ) is None:
            raise KeyError(f"unknown project model: {model_id}")
        value = json_value(dict(payload or {}))
        canonical = json.dumps({
            "project_id": project_id, "model_id": model_id, "runner_kind": runner_kind,
            "payload": value,
        }, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(canonical.encode()).hexdigest()
        configuration_id = "config_" + digest[:24]
        created = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO app_run_configurations VALUES(?,?,?,?,?,?,?,?)",
                (configuration_id, project_id, model_id or None, name, runner_kind,
                 digest, json.dumps(value), created),
            )
        return self.configuration(configuration_id)

    def configuration(self, configuration_id: str) -> RunConfiguration:
        row = self.database.fetch_one(
            "SELECT * FROM app_run_configurations WHERE configuration_id=?", (configuration_id,),
        )
        if row is None:
            raise KeyError(f"unknown run configuration: {configuration_id}")
        return _configuration(row)

    def configurations(self, project_id: str) -> tuple[RunConfiguration, ...]:
        self.projects.get(project_id)
        return tuple(_configuration(row) for row in self.database.fetch_all(
            "SELECT * FROM app_run_configurations WHERE project_id=? ORDER BY created_at,configuration_id",
            (project_id,),
        ))

    def create_run(
        self, configuration_id: str, *, parent_run_id: str = "",
        resume_checkpoint_id: str = "",
    ) -> RunRecord:
        configuration = self.configuration(configuration_id)
        if parent_run_id:
            parent = self.get(parent_run_id)
            if parent.configuration_id != configuration_id:
                raise ValueError("resumed run must use its parent configuration")
        attempt = int(self.database.fetch_one(
            "SELECT COALESCE(MAX(attempt_number),0)+1 FROM app_runs WHERE configuration_id=?",
            (configuration_id,),
        )[0])
        run_id = f"run_{uuid4().hex}"
        relative = Path("runs") / run_id
        project = self.projects.get(configuration.project_id)
        directory = Path(project.root_path) / relative
        directory.mkdir(parents=True, exist_ok=False)
        created = utc_now()
        task = None
        try:
            task = self.tasks.create(configuration.project_id, "simulation-run", {
                "configuration_id": configuration_id, "run_id": run_id,
                "attempt_number": attempt, "parent_run_id": parent_run_id,
                "resume_checkpoint_id": resume_checkpoint_id,
            })
            manifest = {
                "schema_version": 1, "run_id": run_id,
                "configuration": configuration.to_dict(), "attempt_number": attempt,
                "parent_run_id": parent_run_id, "resume_checkpoint_id": resume_checkpoint_id,
                "status": "queued", "created_at": created,
            }
            manifest_path = directory / "manifest.json"
            manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            os.chmod(manifest_path, 0o444)
            with self.database.transaction() as connection:
                connection.execute(
                    "INSERT INTO app_runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (run_id, configuration.project_id, configuration_id, task.task_id,
                     attempt, parent_run_id or None, resume_checkpoint_id, "queued",
                     relative.as_posix(), None, "", "", created),
                )
        except Exception:
            import shutil
            shutil.rmtree(directory)
            if task is not None and self.tasks.get(task.task_id).status.value == "queued":
                self.tasks.transition(
                    task.task_id, "cancelled", message="Run creation rolled back",
                )
            raise
        return self.get(run_id)

    def get(self, run_id: str) -> RunRecord:
        row = self.database.fetch_one("SELECT * FROM app_runs WHERE run_id=?", (run_id,))
        if row is None:
            raise KeyError(f"unknown run: {run_id}")
        return _run(row)

    def list(self, project_id: str) -> tuple[RunRecord, ...]:
        self.projects.get(project_id)
        return tuple(_run(row) for row in self.database.fetch_all(
            "SELECT * FROM app_runs WHERE project_id=? ORDER BY created_at,run_id", (project_id,),
        ))

    def set_status(
        self, run_id: str, status: RunStatus, *, process_id=None,
        started_at: str | None = None, finished_at: str | None = None,
    ) -> RunRecord:
        status = RunStatus(status)
        current = self.get(run_id)
        fields, values = ["status=?"], [status.value]
        if process_id is not None:
            fields.append("process_id=?"); values.append(int(process_id))
        if started_at is not None:
            fields.append("started_at=?"); values.append(started_at)
        if finished_at is not None:
            fields.append("finished_at=?"); values.append(finished_at)
        values.append(run_id)
        with self.database.transaction() as connection:
            connection.execute(f"UPDATE app_runs SET {','.join(fields)} WHERE run_id=?", tuple(values))
        updated = self.get(run_id)
        project = self.projects.get(updated.project_id)
        directory = Path(project.root_path) / updated.run_relative_path
        status_path = directory / "status.json"
        temporary = directory / ".status.json.tmp"
        temporary.write_text(json.dumps(updated.to_dict(), indent=2, sort_keys=True) + "\n")
        temporary.replace(status_path)
        return updated

    def append_event(self, run_id: str, event_type: str, elapsed_seconds: float, payload) -> RunEvent:
        self.get(run_id)
        event_type = nonempty(event_type, "event type")
        payload = json_value(dict(payload or {}))
        created = utc_now()
        with self.database.transaction() as connection:
            sequence = int(connection.execute(
                "SELECT COALESCE(MAX(sequence),-1)+1 FROM app_run_events WHERE run_id=?", (run_id,),
            ).fetchone()[0])
            connection.execute(
                "INSERT INTO app_run_events VALUES(?,?,?,?,?,?)",
                (run_id, sequence, event_type, float(elapsed_seconds), json.dumps(payload), created),
            )
        event = RunEvent(run_id, sequence, event_type, elapsed_seconds, payload, created)
        run = self.get(run_id); project = self.projects.get(run.project_id)
        event_path = Path(project.root_path) / run.run_relative_path / "events.jsonl"
        encoded = (json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode()
        with event_path.open("ab") as stream:
            stream.write(encoded); stream.flush(); os.fsync(stream.fileno())
        return event

    def events(self, run_id: str, *, after_sequence: int = -1, limit: int = 1000) -> tuple[RunEvent, ...]:
        self.get(run_id)
        limit = int(limit)
        if not 1 <= limit <= 10_000:
            raise ValueError("event page limit must lie in [1, 10000]")
        return tuple(RunEvent(
            row["run_id"], row["sequence"], row["event_type"], row["elapsed_seconds"],
            json.loads(row["payload_json"]), row["created_at"],
        ) for row in self.database.fetch_all(
            "SELECT * FROM app_run_events WHERE run_id=? AND sequence>? ORDER BY sequence LIMIT ?",
            (run_id, int(after_sequence), limit),
        ))

    def add_checkpoint(self, run_id: str, relative_path: str, metadata) -> RunCheckpoint:
        run = self.get(run_id)
        project = self.projects.get(run.project_id)
        run_root = (Path(project.root_path) / run.run_relative_path).resolve()
        target = (run_root / relative_path).resolve()
        if run_root not in target.parents or not target.is_file() or target.is_symlink():
            raise ValueError("checkpoint must be a regular file inside its run directory")
        digest = _file_hash(target)
        os.chmod(target, 0o444)
        sequence = int(self.database.fetch_one(
            "SELECT COALESCE(MAX(sequence),-1)+1 FROM app_run_checkpoints WHERE run_id=?", (run_id,),
        )[0])
        checkpoint_id = "checkpoint_" + hashlib.sha256(
            f"{run_id}:{sequence}:{digest}".encode()
        ).hexdigest()[:24]
        created = utc_now(); metadata = json_value(dict(metadata or {}))
        relative = target.relative_to(run_root).as_posix()
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO app_run_checkpoints VALUES(?,?,?,?,?,?,?)",
                (checkpoint_id, run_id, sequence, relative, digest, json.dumps(metadata), created),
            )
        return RunCheckpoint(checkpoint_id, run_id, sequence, relative, digest, metadata, created)

    def checkpoint(self, checkpoint_id: str) -> RunCheckpoint:
        row = self.database.fetch_one(
            "SELECT * FROM app_run_checkpoints WHERE checkpoint_id=?", (checkpoint_id,),
        )
        if row is None:
            raise KeyError(f"unknown run checkpoint: {checkpoint_id}")
        return RunCheckpoint(
            row["checkpoint_id"], row["run_id"], row["sequence"], row["relative_path"],
            row["content_hash"], json.loads(row["metadata_json"]), row["created_at"],
        )

    def checkpoints(self, run_id: str) -> tuple[RunCheckpoint, ...]:
        self.get(run_id)
        return tuple(self.checkpoint(row[0]) for row in self.database.fetch_all(
            "SELECT checkpoint_id FROM app_run_checkpoints WHERE run_id=? ORDER BY sequence",
            (run_id,),
        ))

    def checkpoint_path(self, checkpoint_id: str) -> Path:
        checkpoint = self.checkpoint(checkpoint_id); run = self.get(checkpoint.run_id)
        project = self.projects.get(run.project_id)
        path = Path(project.root_path) / run.run_relative_path / checkpoint.relative_path
        if _file_hash(path) != checkpoint.content_hash:
            raise RuntimeError("checkpoint checksum mismatch")
        return path

    def recover_interrupted(self) -> tuple[RunRecord, ...]:
        rows = self.database.fetch_all(
            "SELECT run_id FROM app_runs WHERE status IN ('starting','running','cancelling')"
        )
        return tuple(self.set_status(row[0], RunStatus.INTERRUPTED, finished_at=utc_now()) for row in rows)


__all__ = ["RunService"]
