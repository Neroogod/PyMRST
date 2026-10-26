"""Stable project navigation tree consumed by desktop and browser shells."""

from dataclasses import dataclass
from typing import Any

from ._common import nonempty


@dataclass(frozen=True, slots=True)
class NavigationItem:
    item_id: str
    label: str
    route: str
    icon: str
    children: tuple["NavigationItem", ...] = ()

    def __post_init__(self) -> None:
        for name in ("item_id", "label", "route", "icon"):
            object.__setattr__(self, name, nonempty(getattr(self, name), name))
        object.__setattr__(self, "children", tuple(self.children))

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id, "label": self.label, "route": self.route,
            "icon": self.icon, "children": [item.to_dict() for item in self.children],
        }


@dataclass(frozen=True, slots=True)
class ProjectNavigation:
    project_id: str
    items: tuple[NavigationItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"project_id": self.project_id,
                "items": [item.to_dict() for item in self.items]}


def default_project_navigation(project_id: str) -> ProjectNavigation:
    base = f"/projects/{project_id}"
    model_children = tuple(
        NavigationItem(name, label, f"{base}/model/{name}", icon)
        for name, label, icon in (
            ("grid", "Grid", "grid"), ("rock", "Rock", "layers"),
            ("fluids", "Fluids", "droplets"), ("initialization", "Initialization", "gauge"),
            ("wells", "Wells", "well"), ("schedule", "Schedule", "calendar"),
        )
    )
    items = (
        NavigationItem("overview", "Overview", base, "home"),
        NavigationItem("model", "Model", f"{base}/model", "cube", model_children),
        NavigationItem("simulation", "Simulation", f"{base}/simulation", "play"),
        NavigationItem("results", "Results", f"{base}/results", "chart"),
        NavigationItem("experiments", "Experiments", f"{base}/experiments", "flask"),
        NavigationItem("agents", "Engineering Assistant", f"{base}/agents", "sparkles"),
        NavigationItem("reports", "Reports", f"{base}/reports", "document"),
    )
    return ProjectNavigation(project_id, items)


__all__ = ["NavigationItem", "ProjectNavigation", "default_project_navigation"]
