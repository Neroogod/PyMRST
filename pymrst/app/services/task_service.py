"""Durable task lifecycle independent of any execution backend."""

from __future__ import annotations

import json
from uuid import uuid4

from ..database.models import AppDatabase
from ..schemas._common import json_value, utc_now
from ..schemas.task import TaskRecord, TaskStatus
from .notification_service import NotificationService


_TRANSITIONS = {
    TaskStatus.QUEUED: {TaskStatus.RUNNING, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {
        TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED,
        TaskStatus.INTERRUPTED,
    },
    TaskStatus.INTERRUPTED: {TaskStatus.QUEUED, TaskStatus.CANCELLED},
    TaskStatus.COMPLETED: set(), TaskStatus.FAILED: set(), TaskStatus.CANCELLED: set(),
}


def _decode(value):
    return None if value is None else json.loads(value)


def _from_row(row) -> TaskRecord:
    return TaskRecord(
        row["task_id"], row["project_id"], row["kind"], row["status"],
        row["progress"], row["message"], json.loads(row["request_json"]),
        _decode(row["result_json"]), _decode(row["error_json"]),
        row["created_at"], row["updated_at"], bool(row["cancellation_requested"]),
    )


class TaskService:
    def __init__(
        self, database: AppDatabase,
        notifications: NotificationService | None = None,
    ) -> None:
        self.database = database
        self.notifications = notifications

    def create(self, project_id: str, kind: str, request=None) -> TaskRecord:
        if self.database.fetch_one(
            "SELECT 1 FROM app_projects WHERE project_id=?", (str(project_id),),
        ) is None:
            raise KeyError(f"unknown project: {project_id}")
        now = utc_now()
        record = TaskRecord(
            f"task_{uuid4().hex}", project_id, kind, TaskStatus.QUEUED,
            0.0, "Queued", json_value(request or {}), None, None, now, now, False,
        )
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO app_tasks VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    record.task_id, record.project_id, record.kind, record.status.value,
                    record.progress, record.message, json.dumps(record.request), None, None,
                    0, record.created_at, record.updated_at,
                ),
            )
        return record

    def get(self, task_id: str) -> TaskRecord:
        row = self.database.fetch_one(
            "SELECT * FROM app_tasks WHERE task_id=?", (str(task_id),),
        )
        if row is None:
            raise KeyError(f"unknown task: {task_id}")
        return _from_row(row)

    def list(self, *, project_id=None, statuses=()) -> tuple[TaskRecord, ...]:
        conditions, parameters = [], []
        if project_id is not None:
            conditions.append("project_id=?"); parameters.append(str(project_id))
        statuses = tuple(TaskStatus(item) for item in statuses)
        if statuses:
            conditions.append("status IN (" + ",".join("?" for _ in statuses) + ")")
            parameters.extend(item.value for item in statuses)
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        return tuple(_from_row(row) for row in self.database.fetch_all(
            "SELECT * FROM app_tasks" + where + " ORDER BY created_at, task_id",
            tuple(parameters),
        ))

    def transition(
        self, task_id: str, status, *, progress=None, message=None,
        result=None, error=None,
    ) -> TaskRecord:
        current = self.get(task_id)
        status = TaskStatus(status)
        if status not in _TRANSITIONS[current.status]:
            raise ValueError(
                f"illegal task transition {current.status.value} -> {status.value}"
            )
        next_progress = current.progress if progress is None else float(progress)
        if status is TaskStatus.COMPLETED:
            next_progress = 1.0
        if next_progress < current.progress and status not in {TaskStatus.QUEUED}:
            raise ValueError("task progress cannot decrease")
        if status is TaskStatus.FAILED and error is None:
            raise ValueError("failed task transition requires error details")
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE app_tasks SET status=?,progress=?,message=?,result_json=?,"
                "error_json=?,updated_at=? WHERE task_id=?",
                (
                    status.value, next_progress,
                    current.message if message is None else str(message),
                    None if result is None else json.dumps(json_value(result)),
                    None if error is None else json.dumps(json_value(error)),
                    now, task_id,
                ),
            )
            if self.notifications is not None and status in {
                TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED,
                TaskStatus.INTERRUPTED,
            }:
                severity = {
                    TaskStatus.COMPLETED: "success", TaskStatus.FAILED: "error",
                    TaskStatus.CANCELLED: "warning", TaskStatus.INTERRUPTED: "warning",
                }[status]
                self.notifications.create(
                    severity, f"Task {status.value}",
                    current.message if message is None else str(message),
                    project_id=current.project_id, task_id=current.task_id,
                    connection=connection,
                )
        updated = self.get(task_id)
        return updated

    def update_progress(self, task_id: str, progress: float, message: str) -> TaskRecord:
        current = self.get(task_id)
        if current.status is not TaskStatus.RUNNING:
            raise ValueError("only running tasks accept progress updates")
        progress = float(progress)
        if not current.progress <= progress < 1:
            raise ValueError("running-task progress must be monotonic and below 1")
        now = utc_now()
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE app_tasks SET progress=?,message=?,updated_at=? WHERE task_id=?",
                (progress, str(message), now, task_id),
            )
        return self.get(task_id)

    def request_cancellation(self, task_id: str) -> TaskRecord:
        current = self.get(task_id)
        if current.status is TaskStatus.QUEUED:
            return self.transition(task_id, TaskStatus.CANCELLED, message="Cancelled before start")
        if current.status is not TaskStatus.RUNNING:
            raise ValueError("only queued or running tasks can be cancelled")
        with self.database.transaction() as connection:
            connection.execute(
                "UPDATE app_tasks SET cancellation_requested=1,updated_at=? WHERE task_id=?",
                (utc_now(), task_id),
            )
        return self.get(task_id)

    def recover_interrupted(self) -> tuple[TaskRecord, ...]:
        running = self.list(statuses=(TaskStatus.RUNNING,))
        return tuple(self.transition(
            item.task_id, TaskStatus.INTERRUPTED,
            message="Application restarted while task was running",
        ) for item in running)


__all__ = ["TaskService"]
