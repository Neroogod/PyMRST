"""Persistent application notification contracts."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from ._common import nonempty


class NotificationSeverity(StrEnum):
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class NotificationRecord:
    notification_id: str
    severity: NotificationSeverity
    title: str
    message: str
    created_at: str
    project_id: str = ""
    task_id: str = ""
    read: bool = False

    def __post_init__(self) -> None:
        for name in ("notification_id", "title", "message", "created_at"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        object.__setattr__(self, "severity", NotificationSeverity(self.severity))

    def to_dict(self) -> dict[str, Any]:
        value = {name: getattr(self, name) for name in self.__slots__}
        value["severity"] = self.severity.value
        return value


__all__ = ["NotificationRecord", "NotificationSeverity"]
