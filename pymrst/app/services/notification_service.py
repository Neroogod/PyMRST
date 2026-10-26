"""Persistent notifications for tasks, validation, and user actions."""

from __future__ import annotations

from uuid import uuid4

from ..database.models import AppDatabase
from ..schemas._common import utc_now
from ..schemas.notification import NotificationRecord, NotificationSeverity


def _from_row(row) -> NotificationRecord:
    return NotificationRecord(
        row["notification_id"], row["severity"], row["title"], row["message"],
        row["created_at"], row["project_id"], row["task_id"], bool(row["is_read"]),
    )


class NotificationService:
    def __init__(self, database: AppDatabase) -> None:
        self.database = database

    def create(
        self, severity, title, message, *, project_id="", task_id="",
        connection=None,
    ) -> NotificationRecord:
        record = NotificationRecord(
            f"notification_{uuid4().hex}", NotificationSeverity(severity),
            title, message, utc_now(), str(project_id), str(task_id), False,
        )
        def insert(owner):
            owner.execute(
                "INSERT INTO app_notifications VALUES(?,?,?,?,?,?,?,?)",
                (
                    record.notification_id, record.project_id, record.task_id,
                    record.severity.value, record.title, record.message,
                    int(record.read), record.created_at,
                ),
            )
        if connection is None:
            with self.database.transaction() as owner:
                insert(owner)
        else:
            insert(connection)
        return record

    def list(self, *, project_id=None, unread_only=False):
        conditions, parameters = [], []
        if project_id is not None:
            conditions.append("project_id=?"); parameters.append(str(project_id))
        if unread_only:
            conditions.append("is_read=0")
        where = " WHERE " + " AND ".join(conditions) if conditions else ""
        rows = self.database.fetch_all(
            "SELECT * FROM app_notifications" + where
            + " ORDER BY created_at DESC, notification_id", tuple(parameters),
        )
        return tuple(_from_row(row) for row in rows)

    def mark_read(self, notification_id: str, *, read=True) -> NotificationRecord:
        with self.database.transaction() as connection:
            cursor = connection.execute(
                "UPDATE app_notifications SET is_read=? WHERE notification_id=?",
                (int(bool(read)), str(notification_id)),
            )
            if cursor.rowcount != 1:
                raise KeyError(f"unknown notification: {notification_id}")
        row = self.database.fetch_one(
            "SELECT * FROM app_notifications WHERE notification_id=?",
            (str(notification_id),),
        )
        return _from_row(row)


__all__ = ["NotificationService"]
