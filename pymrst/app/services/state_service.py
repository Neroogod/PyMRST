"""Transactional application and workspace state."""

from __future__ import annotations

import json

from ..database.models import AppDatabase
from ..schemas._common import json_value, utc_now
from ..schemas.workspace import WorkspaceState


class ApplicationStateService:
    WORKSPACE_KEY = "workspace"

    def __init__(self, database: AppDatabase) -> None:
        self.database = database

    def set(self, key: str, value) -> None:
        key = str(key).strip()
        if not key:
            raise ValueError("application setting key cannot be empty")
        encoded = json.dumps(json_value(value), sort_keys=True, allow_nan=False)
        with self.database.transaction() as connection:
            connection.execute(
                "INSERT INTO app_settings(setting_key,value_json,updated_at) VALUES(?,?,?) "
                "ON CONFLICT(setting_key) DO UPDATE SET value_json=excluded.value_json,"
                "updated_at=excluded.updated_at",
                (key, encoded, utc_now()),
            )

    def get(self, key: str, default=None):
        row = self.database.fetch_one(
            "SELECT value_json FROM app_settings WHERE setting_key=?", (str(key),),
        )
        return default if row is None else json.loads(row["value_json"])

    def workspace(self) -> WorkspaceState:
        value = self.get(self.WORKSPACE_KEY)
        return WorkspaceState() if value is None else WorkspaceState.from_dict(value)

    def save_workspace(self, state: WorkspaceState) -> WorkspaceState:
        if not isinstance(state, WorkspaceState):
            raise TypeError("workspace state service requires WorkspaceState")
        if state.active_project_id and self.database.fetch_one(
            "SELECT 1 FROM app_projects WHERE project_id=?", (state.active_project_id,),
        ) is None:
            raise ValueError("workspace state references an unknown project")
        self.set(self.WORKSPACE_KEY, state.to_dict())
        return state


__all__ = ["ApplicationStateService"]
