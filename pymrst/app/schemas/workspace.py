"""Persistent cross-workspace UI selection and layout state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from ._common import json_value


@dataclass(frozen=True, slots=True)
class WorkspaceState:
    active_project_id: str = ""
    route: str = "/"
    selection: Mapping[str, Any] = field(default_factory=dict)
    bottom_panel: str = "tasks"
    bottom_panel_open: bool = True
    expert_mode: bool = False

    def __post_init__(self) -> None:
        route = str(self.route).strip()
        panel = str(self.bottom_panel).strip()
        if not route.startswith("/") or not panel:
            raise ValueError("workspace route and bottom panel are invalid")
        object.__setattr__(self, "active_project_id", str(self.active_project_id).strip())
        object.__setattr__(self, "route", route)
        object.__setattr__(self, "bottom_panel", panel)
        object.__setattr__(self, "selection", json_value(self.selection, "$.selection"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_project_id": self.active_project_id, "route": self.route,
            "selection": self.selection, "bottom_panel": self.bottom_panel,
            "bottom_panel_open": self.bottom_panel_open,
            "expert_mode": self.expert_mode,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "WorkspaceState":
        fields = {
            "active_project_id", "route", "selection", "bottom_panel",
            "bottom_panel_open", "expert_mode",
        }
        if set(value) != fields:
            raise ValueError("invalid WorkspaceState fields")
        return cls(**value)


__all__ = ["WorkspaceState"]
