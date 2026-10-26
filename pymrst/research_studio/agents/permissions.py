"""Explicit authority and resource boundaries for engineering agents."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Mapping

from ._serialization import require_fields
from .policies.resource_limits import ExecutionBudget


class AgentPermission(IntEnum):
    INSPECT = 0
    ANALYZE = 1
    CREATE_VARIANT = 2
    RUN_SIMULATION = 3
    RUN_ENSEMBLE_OR_REMOTE = 4
    PUBLISH = 5


@dataclass(frozen=True, slots=True)
class PermissionGrant:
    maximum: AgentPermission = AgentPermission.INSPECT
    budget: ExecutionBudget = ExecutionBudget()
    allowed_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "maximum", AgentPermission(self.maximum))
        tools = tuple(str(tool).strip() for tool in self.allowed_tools)
        if any(not tool for tool in tools):
            raise ValueError("allowed tool names cannot be empty")
        if len(set(tools)) != len(tools):
            raise ValueError("allowed tool names must be unique")
        object.__setattr__(self, "allowed_tools", tools)

    def permits(self, permission: AgentPermission, tool: str | None = None) -> bool:
        if AgentPermission(permission) > self.maximum:
            return False
        return tool is None or not self.allowed_tools or tool in self.allowed_tools

    def to_dict(self) -> dict[str, Any]:
        return {
            "maximum": self.maximum.name.lower(),
            "budget": self.budget.to_dict(),
            "allowed_tools": list(self.allowed_tools),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PermissionGrant":
        data = require_fields(
            value,
            required={"maximum", "budget", "allowed_tools"},
            name="PermissionGrant",
        )
        maximum = data["maximum"]
        if isinstance(maximum, str):
            try:
                maximum = AgentPermission[maximum.upper()]
            except KeyError as error:
                raise ValueError(f"unknown agent permission {maximum!r}") from error
        return cls(
            maximum=maximum,
            budget=ExecutionBudget.from_dict(data["budget"]),
            allowed_tools=tuple(data["allowed_tools"]),
        )


__all__ = ["AgentPermission", "ExecutionBudget", "PermissionGrant"]
